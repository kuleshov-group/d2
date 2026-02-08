import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
import os
from SFT_AO.anyorder_sft_trainer import *
import torch.distributed as dist
import random
import numpy as np
import custom_llada


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="any-order causal LLaDA", help="Name of the pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-ao-gsm8k", help="Job Name")
    parser.add_argument("--train_data", type=str, default="gsm8k", help="Path to training data")
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--block_length", type=int, default=32)

    return parser.parse_args()


# Model loading with LoRA integration
def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    # Load model
    ao_model = custom_llada.LLaDAModelLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Freeze all params except q_proj/k_proj/v_proj
    for name, param in ao_model.named_parameters():
        if any(t in name for t in ("q_proj", "k_proj", "v_proj")):
            param.requires_grad = True
        else:
            param.requires_grad = False

    ao_model = ao_model.to(torch.bfloat16)  # Cast fp32 lora params to bf16

    trainable = sum(p.numel() for p in ao_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ao_model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.4f}%)")

    return tokenizer, ao_model


# Dataset loading
def load_data(args, tokenizer):
    if args.train_data == 'gsm8k':
        train_data, eval_data = preprocess_dataset_gsm8k(args.data_path, tokenizer, args.block_length)
    elif args.train_data == 'math':
        train_data, eval_data = preprocess_dataset_math(args.data_path, tokenizer, args.block_length)
    else:
        raise NotImplementedError
    
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, eval=True)
    return train_dataset, eval_dataset


# Training setup
def train_model(args, tokenizer, ao_model):
    # Load dataset
    train_dataset, eval_dataset = load_data(args, tokenizer)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=2,
        save_steps=1000,
        save_total_limit=200,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=ao_model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    init_seed(42)
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    tokenizer, ao_model = load_model_and_tokenizer(args)

    # Train the model
    train_model(args, tokenizer, ao_model)
