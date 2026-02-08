import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
import os

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class diffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    def _get_train_sampler(self, train_dataset=None):
        # Compatibility shim for TRL/Transformers signature mismatch.
        if train_dataset is None:
            train_dataset = self.train_dataset
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        position_ids = inputs["position_ids"]
        completion_mask =  inputs["completion_mask"] 

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        input_ids = input_ids.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)
        per_token_logps = self._get_per_token_logps(model, input_ids, position_ids, logits_to_keep)
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        advantages = inputs["advantages"]
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-6)
            mean_kl_metric = self.accelerator.gather_for_metrics(mean_kl)
            self._metrics[mode]["kl"].append(mean_kl_metric.mean().item())

        loss = (per_token_loss * completion_mask).sum() / (completion_mask.sum() + 1e-6)

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / (completion_mask.sum() + 1e-6)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _preprocess_attention_mask(self, attention_mask, dtype=torch.float):
        min_dtype = torch.finfo(dtype).min
        attention_mask = torch.where(
            (attention_mask == 0.0).bool(),  # type: ignore
            min_dtype,
            0.0,
        ).to(dtype)
        return attention_mask

    @torch.no_grad()
    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            prompt_len = prompt.shape[1]
            x = prompt.clone()
            batch_size = x.shape[0]
            dtype = model.dtype
            # initial attention_bias  
            attention_bias = torch.full(
                (batch_size, 1, prompt_len, prompt_len),
                1., dtype=dtype
            ).to(model.device)
            # initial position_ids
            position_ids = torch.arange(prompt_len, device=model.device).unsqueeze(0).expand(batch_size, -1).clone()
            # initial select_index
            select_index_1 = torch.zeros(batch_size, dtype=torch.long, device=model.device)
            select_index_2 = torch.zeros(batch_size, dtype=torch.long, device=model.device)

            num_blocks = gen_length // block_length
            for i in range(num_blocks):
                # change x
                mask_tensor = torch.full((x.shape[0], block_length), mask_id).to(prompt.device)
                x = torch.cat([x, mask_tensor], dim=1)
                # change attention_bias
                curr_seq_len = attention_bias.shape[-1]
                attention_bias_1 = torch.full((batch_size, 1, block_length, curr_seq_len), 1., dtype=dtype).to(model.device)
                attention_bias = torch.cat([attention_bias, attention_bias_1], dim=2)
                attention_bias_2 = torch.full((batch_size, 1, curr_seq_len + block_length, block_length), 0., dtype=dtype).to(model.device)
                attention_bias = torch.cat([attention_bias, attention_bias_2], dim=3)
                attention_bias[..., curr_seq_len + torch.arange(block_length), curr_seq_len + torch.arange(block_length)] = 1.
                # change position_ids
                new_position_ids = torch.arange(curr_seq_len, curr_seq_len + block_length).to(model.device).unsqueeze(0).expand(batch_size, -1).clone()
                position_ids = torch.cat([position_ids, new_position_ids], dim=1)

                for i in range(block_length // 2):
                    mask_index = x == mask_id

                    attn_bias = self._preprocess_attention_mask(attention_bias, dtype=dtype)
                    logits = model(input_ids=x, attention_bias=attn_bias, position_ids=position_ids).logits

                    # Apply Gumbel noise for sampling
                    logits_with_noise = self.add_gumbel_noise(logits, temperature, dtype=dtype)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    # Handle remasking strategy
                    if remasking == "low_confidence":
                        # Use float32 instead of float64 for better performance
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    elif remasking == "random":
                        x0_p = torch.rand(x0.shape, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    # Select tokens to transfer based on confidence
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        num_tokens = 2
                        if num_tokens > 0:
                            _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            transfer_index[j, select_indices] = True
                            select_index_1[j] = min(select_indices[0].item(), select_indices[1].item())
                            select_index_2[j] = max(select_indices[0].item(), select_indices[1].item())
                    
                    x[transfer_index] = x0[transfer_index]

                    # swap the decoded token and the nearest token
                    curr_seq_len += 2
                    attention_bias[..., curr_seq_len - 1:, curr_seq_len - 2] = 1.
                    attention_bias[..., curr_seq_len:, curr_seq_len - 1] = 1.
                    
                    for j in range(batch_size):
                        idx1 = select_index_1[j].item()
                        idx2 = select_index_2[j].item()
                        # token 1
                        tmp1, tmp2 = position_ids[j, curr_seq_len - 2], position_ids[j, idx1]
                        position_ids[j, idx1], position_ids[j, curr_seq_len - 2] = tmp1.clone(), tmp2.clone()
                        tmp1, tmp2 = x[j, curr_seq_len - 2], x[j, idx1]
                        x[j, idx1], x[j, curr_seq_len - 2] = tmp1.clone(), tmp2.clone()
                        # token 2
                        tmp1, tmp2 = position_ids[j, curr_seq_len - 1], position_ids[j, idx2]
                        position_ids[j, idx2], position_ids[j, curr_seq_len - 1] = tmp1.clone(), tmp2.clone()
                        tmp1, tmp2 = x[j, curr_seq_len - 1], x[j, idx2]
                        x[j, idx2], x[j, curr_seq_len - 1] = tmp1.clone(), tmp2.clone()
            
            return x, position_ids

    def get_logits(self, model, x, position_ids, gen_length, mask_id):
        # get parameters
        batch_size = x.shape[0]
        dtype = torch.bfloat16
        prompt_len = x.shape[1] - gen_length
        # start to compute logits
        x[:, -gen_length:] = mask_id
        attention_bias = torch.full(
            (batch_size, 1, x.shape[1], x.shape[1]),
            0, dtype=dtype
        ).to(model.device)
        attention_bias[..., :prompt_len] = 1.
        attention_bias[..., prompt_len:prompt_len + gen_length, prompt_len:prompt_len + gen_length] = torch.tril(torch.ones(gen_length, gen_length, dtype=attention_bias.dtype, device=attention_bias.device))
        attn_bias = self._preprocess_attention_mask(attention_bias, dtype=dtype)
        logits = model(input_ids=x, attention_bias=attn_bias, position_ids=position_ids).logits
        
        return logits

    def _get_per_token_logps(self, model, input_ids, position_id, logits_to_keep):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()

        # applying masks
        all_inputs = [input_ids[iter_idx] for iter_idx in range(input_ids.shape[0])]
        inputs = torch.cat(all_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        all_position_ids = [position_id[iter_idx] for iter_idx in range(input_ids.shape[0])]
        position_ids = torch.cat(all_position_ids, dim=0)

        # Get model predictions for the combined batch
        logits = self.get_logits(model, inputs, position_ids, gen_length=logits_to_keep, mask_id=self.args.mask_id)  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[:, -logits_to_keep:, :]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = inputs[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del logits
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            prompt_completion_ids, position_ids = self.generate(
                model=unwrapped_model,
                prompt=prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        permutation = torch.argsort(position_ids[:, prompt_length:], dim=1)
        completion_ids_correct_order = torch.gather(completion_ids, 1, permutation)
        is_eos = completion_ids_correct_order == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_correct_order = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_mask = torch.gather(completion_mask_correct_order, 1, torch.argsort(permutation, dim=1))

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                position_ids_expanded = position_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, position_ids_expanded, logits_to_keep,
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, position_ids_expanded, logits_to_keep,
                    )
                    all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(torch.gather(completion_ids, 1, permutation), skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if reward_func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]['advantages'].append(torch.abs(advantages).max().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "position_ids": position_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
        }
    

class d2AnyOrderTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    def _get_train_sampler(self, train_dataset=None):
        # Compatibility shim for TRL/Transformers signature mismatch.
        if train_dataset is None:
            train_dataset = self.train_dataset
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        position_ids = inputs["position_ids"]
        completion_mask =  inputs["completion_mask"] 

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        input_ids = input_ids.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)
        per_token_logps = self._get_per_token_logps(model, input_ids, position_ids, logits_to_keep)
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        advantages = inputs["advantages"]
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-6)
            mean_kl_metric = self.accelerator.gather_for_metrics(mean_kl)
            self._metrics[mode]["kl"].append(mean_kl_metric.mean().item())

        loss = (per_token_loss * completion_mask).sum() / (completion_mask.sum() + 1e-6)

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / (completion_mask.sum() + 1e-6)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _preprocess_attention_mask(self, attention_mask, dtype=torch.float):
        min_dtype = torch.finfo(dtype).min
        attention_mask = torch.where(
            (attention_mask == 0.0).bool(),  # type: ignore
            min_dtype,
            0.0,
        ).to(dtype)
        return attention_mask

    @torch.no_grad()
    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            prompt_len = prompt.shape[1]
            x = prompt.clone()
            batch_size = x.shape[0]
            dtype = model.dtype
            # initial attention_bias  
            attention_bias = torch.full(
                (batch_size, 1, prompt_len, prompt_len),
                1., dtype=dtype
            ).to(model.device)
            # initial position_ids
            position_ids = torch.arange(prompt_len, device=model.device).unsqueeze(0).expand(batch_size, -1).clone()
            # initial select_index
            select_index_1 = torch.zeros(batch_size, dtype=torch.long, device=model.device)
            select_index_2 = torch.zeros(batch_size, dtype=torch.long, device=model.device)

            num_blocks = gen_length // block_length
            for i in range(num_blocks):
                # change x
                mask_tensor = torch.full((x.shape[0], block_length), mask_id).to(prompt.device)
                x = torch.cat([x, mask_tensor], dim=1)
                # change attention_bias
                curr_seq_len = attention_bias.shape[-1]
                attention_bias_1 = torch.full((batch_size, 1, block_length, curr_seq_len), 1., dtype=dtype).to(model.device)
                attention_bias = torch.cat([attention_bias, attention_bias_1], dim=2)
                attention_bias_2 = torch.full((batch_size, 1, curr_seq_len + block_length, block_length), 0., dtype=dtype).to(model.device)
                attention_bias = torch.cat([attention_bias, attention_bias_2], dim=3)
                attention_bias[..., curr_seq_len + torch.arange(block_length), curr_seq_len + torch.arange(block_length)] = 1.
                # change position_ids
                new_position_ids = torch.arange(curr_seq_len, curr_seq_len + block_length).to(model.device).unsqueeze(0).expand(batch_size, -1).clone()
                position_ids = torch.cat([position_ids, new_position_ids], dim=1)

                for i in range(block_length // 2):
                    mask_index = x == mask_id

                    attn_bias = self._preprocess_attention_mask(attention_bias, dtype=dtype)
                    logits = model(input_ids=x, attention_bias=attn_bias, position_ids=position_ids).logits

                    # Apply Gumbel noise for sampling
                    logits_with_noise = self.add_gumbel_noise(logits, temperature, dtype=dtype)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    # Handle remasking strategy
                    if remasking == "low_confidence":
                        # Use float32 instead of float64 for better performance
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    elif remasking == "random":
                        x0_p = torch.rand(x0.shape, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    # Select tokens to transfer based on confidence
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        num_tokens = 2
                        if num_tokens > 0:
                            _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            transfer_index[j, select_indices] = True
                            select_index_1[j] = min(select_indices[0].item(), select_indices[1].item())
                            select_index_2[j] = max(select_indices[0].item(), select_indices[1].item())
                    
                    x[transfer_index] = x0[transfer_index]

                    # swap the decoded token and the nearest token
                    curr_seq_len += 2
                    attention_bias[..., curr_seq_len - 1:, curr_seq_len - 2] = 1.
                    attention_bias[..., curr_seq_len:, curr_seq_len - 1] = 1.
                    
                    for j in range(batch_size):
                        idx1 = select_index_1[j].item()
                        idx2 = select_index_2[j].item()
                        # token 1
                        tmp1, tmp2 = position_ids[j, curr_seq_len - 2], position_ids[j, idx1]
                        position_ids[j, idx1], position_ids[j, curr_seq_len - 2] = tmp1.clone(), tmp2.clone()
                        tmp1, tmp2 = x[j, curr_seq_len - 2], x[j, idx1]
                        x[j, idx1], x[j, curr_seq_len - 2] = tmp1.clone(), tmp2.clone()
                        # token 2
                        tmp1, tmp2 = position_ids[j, curr_seq_len - 1], position_ids[j, idx2]
                        position_ids[j, idx2], position_ids[j, curr_seq_len - 1] = tmp1.clone(), tmp2.clone()
                        tmp1, tmp2 = x[j, curr_seq_len - 1], x[j, idx2]
                        x[j, idx2], x[j, curr_seq_len - 1] = tmp1.clone(), tmp2.clone()
            
            return x, position_ids

    def get_logits(self, model, x, position_ids, gen_length, mask_id):
        # get parameters
        batch_size = x.shape[0]
        dtype = torch.bfloat16
        prompt_len = x.shape[1] - gen_length
        # start to compute logits
        mask_delta = torch.full((batch_size, gen_length), mask_id).to(x.device)
        x_2L = torch.cat([x, mask_delta], dim=1)
        mask_position_ids = position_ids.clone()[:, -gen_length:]
        position_ids_2L = torch.cat([position_ids, mask_position_ids], dim=1)
        attention_bias_2L = torch.full(
            (batch_size, 1, x_2L.shape[1], x_2L.shape[1]),
            0, dtype=dtype
        ).to(model.device)
        attention_bias_2L[..., :prompt_len] = 1.
        attention_bias_2L[..., prompt_len:prompt_len + gen_length, prompt_len:prompt_len + gen_length] = torch.tril(torch.ones(gen_length, gen_length, dtype=attention_bias_2L.dtype, device=attention_bias_2L.device))
        pair_rows = torch.arange(gen_length, device=attention_bias_2L.device) // 2
        pair_cols = torch.arange(gen_length, device=attention_bias_2L.device) // 2
        pair_mask = (pair_rows[:, None] > pair_cols[None, :]).to(attention_bias_2L.dtype)
        attention_bias_2L[
            ...,
            prompt_len + gen_length:prompt_len + 2 * gen_length,
            prompt_len:prompt_len + gen_length,
        ] = pair_mask
        #attention_bias_2L[..., prompt_len + gen_length + torch.arange(gen_length), prompt_len + torch.arange(gen_length)] = 0.
        attention_bias_2L[..., prompt_len + gen_length + torch.arange(gen_length), prompt_len + gen_length + torch.arange(gen_length)] = 1.
        attn_bias_2L = self._preprocess_attention_mask(attention_bias_2L, dtype=dtype)
        logits = model(input_ids=x_2L, attention_bias=attn_bias_2L, position_ids=position_ids_2L).logits
        
        return logits

    def _get_per_token_logps(self, model, input_ids, position_id, logits_to_keep):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()

        # applying masks
        all_inputs = [input_ids[iter_idx] for iter_idx in range(input_ids.shape[0])]
        inputs = torch.cat(all_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        all_position_ids = [position_id[iter_idx] for iter_idx in range(input_ids.shape[0])]
        position_ids = torch.cat(all_position_ids, dim=0)

        # Get model predictions for the combined batch
        logits = self.get_logits(model, inputs, position_ids, gen_length=logits_to_keep, mask_id=self.args.mask_id)  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[:, -logits_to_keep:, :]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = inputs[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del logits
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            prompt_completion_ids, position_ids = self.generate(
                model=unwrapped_model,
                prompt=prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        permutation = torch.argsort(position_ids[:, prompt_length:], dim=1)
        completion_ids_correct_order = torch.gather(completion_ids, 1, permutation)
        is_eos = completion_ids_correct_order == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_correct_order = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_mask = torch.gather(completion_mask_correct_order, 1, torch.argsort(permutation, dim=1))

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                position_ids_expanded = position_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, position_ids_expanded, logits_to_keep,
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, position_ids_expanded, logits_to_keep,
                    )
                    all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(torch.gather(completion_ids, 1, permutation), skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if reward_func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]['advantages'].append(torch.abs(advantages).max().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "position_ids": position_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
        }
