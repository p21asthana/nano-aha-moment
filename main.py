import os
import argparse
import torch
import torch.distributed as dist
import deepspeed
import wandb
from transformers import AutoTokenizer
import numpy as np
import json # Added for config saving

# Project Modules
import config as cfg # Use cfg prefix for clarity
from data_utils import load_and_preprocess_data
from models import initialize_policy_and_reference_models, initialize_inference_engine
from trainer import train_loop
from utils import find_last_checkpoint # Keep find_last_checkpoint here

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train R1 model with PPO and DeepSpeed")
    parser.add_argument("--model_name", type=str, default=cfg.DEFAULT_MODEL_NAME, help="Model name/path")
    parser.add_argument("--learning_rate", type=float, default=cfg.DEFAULT_LEARNING_RATE, help="Learning rate for training")
    parser.add_argument("--kl_coeff", type=float, default=cfg.DEFAULT_KL_COEFF, help="KL coefficient for PPO")
    parser.add_argument("--temperature", type=float, default=cfg.DEFAULT_TEMPERATURE, help="Temperature for sampling")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by DeepSpeed)")
    # Add other relevant args from config if needed, e.g., batch sizes, iterations
    parser.add_argument("--num_iterations", type=int, default=cfg.DEFAULT_NUM_ITERATIONS, help="Total number of training iterations")
    parser.add_argument("--episodes_per_iteration", type=int, default=cfg.DEFAULT_EPISODES_PER_ITERATION, help="Global batch size (episodes per iteration)")
    parser.add_argument("--generations_per_sample", type=int, default=cfg.DEFAULT_GENERATIONS_PER_SAMPLE, help="Number of responses per prompt")
    parser.add_argument("--per_device_batch_size", type=int, default=cfg.DEFAULT_PER_DEVICE_BATCH_SIZE, help="Micro-batch size per GPU")
    parser.add_argument("--no_vllm", action="store_true", help="If set, skip initializing vLLM inference engine and use policy model for generation")

    args = parser.parse_args()
    return args

def build_deepspeed_configs(args, world_size, gradient_accumulation_steps):
    """Builds the dynamic DeepSpeed config dictionaries."""
    # Policy Model Config (ZeRO Stage 3)
    policy_config = {
        "bf16": {"enabled": True},
        "zero_optimization": cfg.DEFAULT_ZERO_STAGE_3_CONFIG,
        "optimizer": {
            **cfg.DEFAULT_OPTIMIZER_PARAMS, # Merge default optimizer params
            "params": {
                "lr": args.learning_rate, # Set LR from args
                **cfg.DEFAULT_OPTIMIZER_PARAMS["params"] # Merge other default params
            }
        },
        "scheduler": {
            **cfg.DEFAULT_SCHEDULER_PARAMS, # Merge default scheduler params
             "params": {
                 **cfg.DEFAULT_SCHEDULER_PARAMS["params"],
                 "warmup_max_lr": args.learning_rate, # Set max LR from args
             }
        },
        "gradient_clipping": 1.0,
        "train_batch_size": args.episodes_per_iteration,
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_allow_untested_optimizer": True,
        "fp32_residual_connection": False,
    }

    # Reference Model Config (ZeRO Stage 3, no optimizer state)
    ref_config = {
        "bf16": {"enabled": True},
        "zero_optimization": cfg.DEFAULT_ZERO_REF_STAGE_3_CONFIG,
        "train_batch_size": args.episodes_per_iteration,
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
    }
    return policy_config, ref_config

def main():
    args = parse_arguments()

    # --- Distributed Setup ---
    deepspeed.init_distributed()
    local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_rank_0 = local_rank == 0

    if is_rank_0:
        print(f"Starting distributed training with world size: {world_size}")
        os.environ["HF_HOME"] = str(cfg.HF_HOME)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Configuration --- #
    model_name = args.model_name
    model_chat_name = model_name + cfg.DEFAULT_MODEL_CHAT_NAME_SUFFIX

    # Calculate gradient accumulation steps
    if world_size > 0 and args.per_device_batch_size > 0:
        global_micro_batch_size = args.per_device_batch_size * world_size
        if args.episodes_per_iteration % global_micro_batch_size == 0:
            gradient_accumulation_steps = args.episodes_per_iteration // global_micro_batch_size
        else:
            gradient_accumulation_steps = max(1, args.episodes_per_iteration // global_micro_batch_size)
            if is_rank_0:
                print(f"Warning: episodes_per_iteration not divisible by micro_batch_size * world_size. Setting GAS to {gradient_accumulation_steps}")
    else:
        gradient_accumulation_steps = 1

    policy_ds_config, ref_ds_config = build_deepspeed_configs(args, world_size, gradient_accumulation_steps)

    run_name = f"ds_{model_name.split('/')[-1]}_temp{args.temperature}_kl{args.kl_coeff}_lr{args.learning_rate}_ws{world_size}"
    exp_dir = cfg.DEFAULT_EXP_DIR_BASE / run_name

    # Consolidate dynamic config elements to pass to trainer
    run_config = {
        "EXP_DIR": exp_dir,
        "GENERATIONS_PER_SAMPLE": args.generations_per_sample,
        "EPISODES_PER_ITERATION": args.episodes_per_iteration,
        "PER_DEVICE_BATCH_SIZE": args.per_device_batch_size,
        "NUM_ITERATIONS": args.num_iterations,
        "TOP_P": cfg.DEFAULT_TOP_P,
        "TOP_K": cfg.DEFAULT_TOP_K,
        "MAX_RESPONSE_TOKENS": cfg.DEFAULT_MAX_RESPONSE_TOKENS,
        # EOS tokens will be added later after tokenizer loading
    }

    if is_rank_0:
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logs and Checkpoints will be saved to: {exp_dir}")
        print(f"Effective Gradient Accumulation Steps: {gradient_accumulation_steps}")
        # Save effective DS config
        # with open(exp_dir / 'policy_deepspeed_config.json', 'w') as f:
        #     json.dump(policy_ds_config, f, indent=2)

    # --- Load Tokenizer --- #
    if is_rank_0:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_chat_name)
    # Need base model tokenizer just for EOS ID
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    eos_token_id = base_tokenizer.eos_token_id
    del base_tokenizer # Free memory
    eos_token = tokenizer.convert_ids_to_tokens(eos_token_id) if eos_token_id is not None else tokenizer.eos_token
    run_config["EOS_TOKEN"] = eos_token
    run_config["EOS_TOKEN_ID"] = eos_token_id

    # --- Load Data (Rank 0) --- #
    train_dataset, test_dataset, dataset_info = None, None, {"train_len": 0, "test_len": 0}
    if is_rank_0:
        print("Loading and preprocessing data...")
        num_proc = max(1, os.cpu_count() // 2)
        train_dataset, test_dataset, dataset_info = load_and_preprocess_data(tokenizer, num_proc)

    # --- Broadcast Data Info --- #
    if world_size > 1:
        dist.broadcast_object_list([dataset_info], src=0)
    if not is_rank_0:
        print(f"Rank {local_rank} received dataset info: {dataset_info}")

    # --- Initialize Models --- #
    if is_rank_0:
        print("Initializing models with DeepSpeed...")
    policy_model, ref_model, optimizer, scheduler = initialize_policy_and_reference_models(
        model_name=model_name,
        deepspeed_policy_config=policy_ds_config,
        deepspeed_ref_config=ref_ds_config
    )

    # --- Initialize Inference Engine (Rank 0) --- #
    inference_engine = None
    if not args.no_vllm:
        inference_engine = initialize_inference_engine(
            model_name=model_name,
            tokenizer_name=model_chat_name,
            world_size=world_size,
            rank=local_rank
        )
    else:
        if is_rank_0:
            print("Skipping vLLM initialization (--no_vllm specified). Will use policy model for generation.")

    # --- WandB Setup (Rank 0) --- #
    if is_rank_0:
        print("Initializing WandB...")
        wandb.init(
            project="r1-aha-moment",
            name=run_name,
            config={
                **vars(args), # Log parsed arguments
                "world_size": world_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "policy_deepspeed_config": policy_ds_config, # Log effective config
            },
        )

    # --- Load Checkpoint --- #
    begin_iter = 0
    ckpt_path_str = None
    ckpt_tag = None
    if is_rank_0:
        ckpt_path, found_iter = find_last_checkpoint(exp_dir)
        if ckpt_path:
            load_info = {"ckpt_path": str(ckpt_path.parent), "ckpt_tag": ckpt_path.name, "ckpt_iter": found_iter}
            print(f"Rank 0 identified checkpoint tag '{load_info['ckpt_tag']}' at iteration {load_info['ckpt_iter']}")
        else:
            load_info = {"ckpt_path": None, "ckpt_tag": None, "ckpt_iter": -1}
    else:
        load_info = {"ckpt_path": None, "ckpt_tag": None, "ckpt_iter": -1}

    if world_size > 1:
        dist.broadcast_object_list([load_info], src=0)

    ckpt_path_str = load_info["ckpt_path"]
    ckpt_tag = load_info["ckpt_tag"]
    ckpt_iter = load_info["ckpt_iter"]

    if ckpt_path_str and ckpt_tag:
        print(f"Rank {local_rank} attempting to load checkpoint tag '{ckpt_tag}' from {ckpt_path_str}")
        load_path, _ = policy_model.load_checkpoint(ckpt_path_str, tag=ckpt_tag)
        if load_path is None:
            print(f"Rank {local_rank} WARNING: Failed to load checkpoint tag '{ckpt_tag}'. Starting from scratch.")
            begin_iter = 0
        else:
            # Sync ref model weights
            if hasattr(policy_model.module, 'state_dict'):
                ref_model.load_state_dict(policy_model.module.state_dict())
            else:
                 ref_model.load_state_dict(policy_model.state_dict())
            begin_iter = ckpt_iter + 1
            print(f"Rank {local_rank} successfully loaded checkpoint. Resuming from iteration {begin_iter}")
            # TODO: Add reliable vLLM weight loading from ZeRO-3 checkpoint if needed
            if is_rank_0:
                print("WARNING: Skipping vLLM weight sync from checkpoint due to ZeRO-3 complexity.")
    else:
        print(f"Rank {local_rank} found no checkpoint. Starting from iteration 0.")

    # --- Barrier before training --- #
    if world_size > 1:
        dist.barrier()

    # --- Start Training Loop --- #
    if is_rank_0:
        print("\nStarting training loop...")

    try:
        train_loop(
            args=args,
            config=run_config,
            policy_model=policy_model,
            reference_model=ref_model,
            inference_engine=inference_engine,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            dataset_info=dataset_info,
            begin_iter=begin_iter,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
    except Exception as e:
         print(f"ERROR during training loop on rank {local_rank}: {e}")
         import traceback
         traceback.print_exc()
    finally:
        # --- Cleanup --- #
        if is_rank_0 and wandb.run is not None:
            wandb.finish()
        if world_size > 1:
             # Ensure all processes exit cleanly
             dist.barrier()
             # Consider explicitly destroying process group if needed
             # dist.destroy_process_group()
        print(f"Rank {local_rank} finished.")

if __name__ == "__main__":
    main() 