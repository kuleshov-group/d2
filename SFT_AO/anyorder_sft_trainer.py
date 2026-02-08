import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
import torch.distributed as dist
import re
import jsonlines


def preprocess_attention_mask(attention_mask, dtype=torch.float):
    min_dtype = torch.finfo(dtype).min
    attention_mask = torch.where(
        (attention_mask == 0.0).bool(),  # type: ignore
        min_dtype,
        0.0,
    ).to(dtype)
    return attention_mask


class dLLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_valid_tokens = 0
    def compute_loss(self, ao_model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        input_ids, labels, t, prompt_len, position_ids = inputs.pop("input_ids"), inputs.pop("labels"), inputs.pop("t"), inputs.pop("prompt_length").item(), inputs.pop('position_ids')

        # build the attention_bias
        seq_len = input_ids.shape[1]
        attention_bias = torch.full(
            (1, 1, seq_len, seq_len),
            0, dtype=torch.float
        ).to(ao_model.device)
        # building the mask
        attention_bias[..., :prompt_len] = 1.
        attention_bias[..., prompt_len:prompt_len + t, prompt_len:prompt_len + t] = torch.tril(torch.ones(t, t, dtype=attention_bias.dtype, device=attention_bias.device))
        attention_bias[..., prompt_len + t:seq_len, prompt_len:prompt_len + t] = torch.tril(torch.ones(t, t, dtype=attention_bias.dtype, device=attention_bias.device))
        attention_bias[..., prompt_len + t + torch.arange(t), prompt_len + torch.arange(t)] = 0.
        attention_bias[..., prompt_len + t + torch.arange(t), prompt_len + t + torch.arange(t)] = 1.
        attn_bias = preprocess_attention_mask(attention_bias)

        outputs = ao_model(input_ids=input_ids, attention_bias=attn_bias, position_ids=position_ids)

        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        loss = unscaled_loss.sum() / t

        num_valid_tokens = (labels >= 0).sum().to(ao_model.device)
        if ao_model.training:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(num_valid_tokens, op=dist.ReduceOp.SUM)  # sum over 8 GPUs
            self.total_valid_tokens += num_valid_tokens.item()
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"cum_valid_tokens": self.total_valid_tokens})

        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        return out


def build_block_generation_order(prompt_length, total_length, block_length=32):
    """Create a generation order that only shuffles tokens inside block_length chunks."""
    seq_len = total_length - prompt_length
    if seq_len <= 0:
        return torch.empty(0, dtype=torch.long)

    # Indices of answer tokens follow the prompt positions.
    answer_indices = torch.arange(prompt_length, total_length, dtype=torch.long)
    blocks = torch.split(answer_indices, block_length)
    generation_order = [block[torch.randperm(block.size(0), device=block.device)] for block in blocks]
    return torch.cat(generation_order) if generation_order else torch.empty(0, dtype=torch.long)


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def __call__(self, batch):
        batch = super().__call__(batch)

        prompt_length = batch['prompt_length'].item()
        seq_length = batch['input_ids'].shape[1] - prompt_length
        # prepare position_ids
        block_length = batch.pop('block_length').item()
        generation_order = build_block_generation_order(
                prompt_length,
                seq_length+prompt_length,
                block_length=block_length,
            )
        generation_order_sequence = torch.cat([torch.arange(prompt_length), generation_order])
        position_ids = torch.cat([generation_order_sequence, generation_order])
        batch["position_ids"] = position_ids
        # prepare labels & input_ids
        input_ids = batch.pop('input_ids').squeeze(0)[generation_order_sequence]
        prompt_ids = input_ids[:prompt_length]
        answer_ids = input_ids[prompt_length:]
        labels = torch.cat([prompt_ids, answer_ids, answer_ids])
        noisy_batch = labels.clone()
        mask_indices = torch.arange(labels.shape[0]) >= (prompt_length + seq_length)
        labels[~mask_indices] = -100
        batch['labels'] = labels[None, :]
        noisy_batch[mask_indices] = self.mask_token_id
        batch['input_ids'] = noisy_batch[None, :]
        batch["t"] = seq_length

        return batch


def convert_template(text):
    # 1. Replace the entire SYSTEM block with the Llama-3 start token
    # Matches <role>SYSTEM</role>...content...<|role_end|>
    text = re.sub(
        r'<role>SYSTEM</role>.*?<\|role_end\|>', 
        '<|startoftext|>', 
        text, 
        flags=re.DOTALL
    )
    
    # 2. Convert HUMAN role to Llama-3 User header
    # We strip trailing whitespace (\s*) to enforce the double newline format (\n\n)
    text = re.sub(
        r'<role>HUMAN</role>\s*', 
        '<|start_header_id|>user<|end_header_id|>\n\n', 
        text
    )
    
    # 3. Convert Role End to EOT (End of Turn) id
    text = text.replace('<|role_end|>', '<|eot_id|>')
    
    # 4. Convert ASSISTANT role to Llama-3 Assistant header
    text = re.sub(
        r'<role>ASSISTANT</role>', 
        '<|start_header_id|>assistant<|end_header_id|>\n\n', 
        text
    )
    
    return text


def preprocess_dataset_gsm8k(data_path, tokenizer, block_length=32, test_split=0.05):
    preprocessed_data = []
    with open(data_path, "r") as file:
        reader = jsonlines.Reader(file)
        for obj in reader:
            prompt = convert_template(obj['prompt'])
            answer = obj['answer']
            inputs = prompt + answer
            tokenized_input = tokenizer(inputs, return_tensors="pt")['input_ids']
            tokenized_prompt = tokenizer(prompt, return_tensors="pt")['input_ids']
            input_length = tokenized_input.shape[1]
            prompt_length = tokenized_prompt.shape[1]
            max_length = ((input_length - prompt_length + 1) // block_length + 1) * block_length + prompt_length
            tokenized_input = tokenizer(
                inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
            ).input_ids.squeeze(0)
            assert ((tokenized_input[:prompt_length] - tokenized_prompt[0, :]) != 0).sum() == 0

            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_length": prompt_length,
                    "block_length": block_length,
                }
            )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]

    return train_data, test_data


def preprocess_dataset_math(data_path, tokenizer, block_length=32, test_split=0.05):
    preprocessed_data = []
    with open(data_path, "r") as file:
        reader = jsonlines.Reader(file)
        for obj in reader:
            prompt = convert_template(obj['prompt'])
            answer = obj['answer']
            inputs = prompt + answer
            tokenized_input = tokenizer(inputs, return_tensors="pt")['input_ids']
            tokenized_prompt = tokenizer(prompt, return_tensors="pt")['input_ids']
            input_length = tokenized_input.shape[1]
            prompt_length = tokenized_prompt.shape[1]
            # we filter out the over long prompts and sequences, flexible to change
            if prompt_length > 512:
                continue
            if input_length > 1024 + prompt_length:
                continue
            max_length = ((input_length - prompt_length + 1) // block_length + 1) * block_length + prompt_length
            tokenized_input = tokenizer(
                inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
            ).input_ids.squeeze(0)
            assert ((tokenized_input[:prompt_length] - tokenized_prompt[0, :]) != 0).sum() == 0

            preprocessed_data.append(
                {
                    "input_ids": tokenized_input,
                    "prompt_length": prompt_length,
                    "block_length": block_length,
                }
            )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]

    return train_data, test_data