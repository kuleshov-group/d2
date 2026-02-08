import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


def token_entropy_from_logits(logits, T):
    scaled = logits / T
    log_probs = F.log_softmax(scaled, dim=-1)    # (B, L, V)
    probs = log_probs.exp()                      # (B, L, V)
    H = -(probs * log_probs).sum(dim=-1)         # (B, L)
    return H


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        entropy = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), 0, dtype=torch.float).to(model.device)

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Compute entropy after adding temperature
                entropy_t = token_entropy_from_logits(logits, 1.0)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        transfer_index[j, select_indices] = True
                
                x[transfer_index] = x0[transfer_index]
                entropy[transfer_index] = entropy_t[transfer_index]

        return x, entropy[:, prompt.shape[1]:]


def preprocess_attention_mask(attention_mask, dtype=torch.float):
    min_dtype = torch.finfo(dtype).min
    attention_mask = torch.where(
        (attention_mask == 0.0).bool(),  # type: ignore
        min_dtype,
        0.0,
    ).to(dtype)
    return attention_mask


@torch.no_grad()
def generate_anyorder(
    ao_model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        prompt_len = prompt.shape[1]

        # initial x
        x = prompt.clone()
        # initial attention_bias  
        attention_bias = torch.full(
            (1, 1, prompt_len, prompt_len),
            1., dtype=torch.float
        ).to(ao_model.device)
        # initial position_ids
        position_ids = torch.arange(prompt_len).to(ao_model.device)
        # initial select_index
        select_index_1 = 0
        select_index_2 = 0

        num_blocks = gen_length // block_length
        for i in range(num_blocks):
            # change x
            mask_tensor = torch.full((prompt.shape[0], block_length), mask_id).to(prompt.device)
            x = torch.cat([x, mask_tensor], dim=1)
            # change attention_bias
            curr_seq_len = attention_bias.shape[-1]
            attention_bias_1 = torch.full((1, 1, block_length, curr_seq_len), 1., dtype=torch.float).to(ao_model.device)
            attention_bias = torch.cat([attention_bias, attention_bias_1], dim=2)
            attention_bias_2 = torch.full((1, 1, curr_seq_len + block_length, block_length), 0., dtype=torch.float).to(ao_model.device)
            attention_bias = torch.cat([attention_bias, attention_bias_2], dim=3)
            attention_bias[..., curr_seq_len + torch.arange(block_length), curr_seq_len + torch.arange(block_length)] = 1.
            # change position_ids
            new_position_ids = torch.arange(curr_seq_len, curr_seq_len + block_length).to(ao_model.device)
            position_ids = torch.cat([position_ids, new_position_ids])

            for i in range(block_length // 2):
                mask_index = x == mask_id

                attn_bias = preprocess_attention_mask(attention_bias)
                logits = ao_model(input_ids=x, attention_bias=attn_bias, position_ids=position_ids).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    num_tokens = 2
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        transfer_index[j, select_indices] = True
                        # Here, we use a left-to-right heuristsics, i.e., the token on the left is assigned a smaller sigma value than the token on the right
                        select_index_1 = min(select_indices[0].item(), select_indices[1].item())
                        select_index_2 = max(select_indices[0].item(), select_indices[1].item())
                
                x[transfer_index] = x0[transfer_index]

                # swap the decoded token and the nearest token
                curr_seq_len += 2
                attention_bias[..., curr_seq_len - 1:, curr_seq_len - 2] = 1.
                attention_bias[..., curr_seq_len:, curr_seq_len - 1] = 1.
                
                # token 1
                tmp1, tmp2 = position_ids[curr_seq_len - 2], position_ids[select_index_1]
                position_ids[select_index_1], position_ids[curr_seq_len - 2] = tmp1.clone(), tmp2.clone()
                tmp1, tmp2 = x[..., curr_seq_len - 2], x[..., select_index_1]
                x[..., select_index_1], x[..., curr_seq_len - 2] = tmp1.clone(), tmp2.clone()
                # token 2
                tmp1, tmp2 = position_ids[curr_seq_len - 1], position_ids[select_index_2]
                position_ids[select_index_2], position_ids[curr_seq_len - 1] = tmp1.clone(), tmp2.clone()
                tmp1, tmp2 = x[..., curr_seq_len - 1], x[..., select_index_2]
                x[..., select_index_2], x[..., curr_seq_len - 1] = tmp1.clone(), tmp2.clone()

        permutation = torch.argsort(position_ids)
        return x[0, permutation][None, :]

