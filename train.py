import os
import os.path as osp
import glob
import argparse

import torch
from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

from dataset import load_data, custom_train_collate_fn
from model import load_model, load_tokenizer

import wandb


def main(args):
    tokenizer = load_tokenizer(args)            # Load tokenizer
    train_dataset = load_data(args, tokenizer)  # Load data
    model, peft_config = load_model(args)       # Load model

    # Configure trainer
    trainer = SFTTrainer(
        args=SFTConfig(
            output_dir=args.output_dir,
            dataloader_num_workers=args.num_workers,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_32bit",
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            save_only_model=True,
            torch_compile=args.torch_compile,
            report_to="wandb" if args.wandb_mode == "online" else "none",
            # * SFT arguments
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
            packing=args.packing,
            dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
        ),
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=custom_train_collate_fn if not args.packing else None,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardrails training script")
    # Model arguments
    parser.add_argument("--model_type", type=str, default="qwen2.5-7b")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--cache_dir", type=str, default=None)
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset_name", type=str, default="expguardmix")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--packing", action="store_true")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit_quantization", action="store_true")
    parser.add_argument("--use_8bit_quantization", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--use_peft_lora", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed, deterministic=args.deterministic)

    # Set number of threads for CPU computation
    torch.set_num_threads(1)

    # Set batch size
    world_size = torch.cuda.device_count()
    args.distributed = world_size != 1
    args.train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    args.eval_batch_size = args.per_device_eval_batch_size * world_size
    args.data_dir = osp.join(args.data_dir, args.dataset_name)

    # Set data type
    if args.bf16:
        args.dtype = torch.bfloat16
    elif args.fp16:
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    # Set output directory
    args.group_name = osp.join(args.dataset_name, args.model_type)
    args.run_name = f"BS{args.train_batch_size}_LR{args.learning_rate}_W{args.warmup_ratio}_D{args.weight_decay}_E{args.epochs}_S{args.seed}"
    args.output_dir = osp.join(".checkpoints", args.group_name, args.run_name)

    # Do not overwrite checkpoint files
    if args.do_train and (
        glob.glob(osp.join(args.output_dir, "*.safetensors")) or    # if one checkpoint is saved
        glob.glob(osp.join(args.output_dir, "checkpoint-*"))        # if multiple checkpoints are saved
    ):
        raise FileExistsError(f"Output directory {args.output_dir} already exists.")

    # Set up wandb
    if args.wandb_mode == "online":
        accelerator = Accelerator()
        if accelerator.is_main_process:
            wandb.init(
                project="guardrails",
                group=args.group_name,
                name=args.run_name,
                mode=args.wandb_mode,
            )

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
