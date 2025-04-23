import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM
from vllm import LLM
from typing import Tuple, Optional

# Import config defaults if needed, though typically passed in
# from config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_CHAT_NAME_SUFFIX


def initialize_policy_and_reference_models(
    model_name: str,
    deepspeed_policy_config: dict,
    deepspeed_ref_config: dict
) -> Tuple[deepspeed.DeepSpeedEngine, deepspeed.DeepSpeedEngine, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Loads policy and reference models and initializes DeepSpeed engines."""
    print(f"Loading base model: {model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2", # Requires flash-attn installed
        torch_dtype=torch.bfloat16,
        # device_map should not be used with ZeRO-3
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("Initializing DeepSpeed for Policy Model...")
    policy_model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=policy_model,
        config_params=deepspeed_policy_config,
        model_parameters=policy_model.parameters(),
    )

    print("Initializing DeepSpeed for Reference Model...")
    reference_model_engine, *_ = deepspeed.initialize(
        model=reference_model,
        config_params=deepspeed_ref_config,
        model_parameters=reference_model.parameters(),
    )

    return policy_model_engine, reference_model_engine, optimizer, scheduler

def initialize_inference_engine(
    model_name: str,
    tokenizer_name: str,
    world_size: int,
    rank: int
) -> Optional[LLM]:
    """Initializes the vLLM engine, only on rank 0."""
    if rank != 0:
        return None

    # Calculate GPU memory fraction for vLLM on Rank 0
    # Needs careful tuning based on model size and training memory usage
    # Reduce this value as Rank 0 GPU is shared with training
    vllm_gpu_util = 0.25 # Default: Use 25% of Rank 0 GPU for vLLM
    print(f"Initializing vLLM on Rank 0 with gpu_memory_utilization={vllm_gpu_util:.2f}")

    try:
        # Consider using enforce_eager=True if issues occur with ZeRO-3 weight loading
        inference_engine = LLM(
            model=model_name,
            tokenizer=tokenizer_name, # Pass tokenizer name/path
            gpu_memory_utilization=vllm_gpu_util,
            max_model_len=2048, # Example value, adjust if needed
            dtype="bfloat16", # Match model dtype
            # swap_space=4, # Optional: Adjust based on CPU RAM
            # enforce_eager=True,
            # tensor_parallel_size=1, # Ensure TP=1 as only rank 0 uses it
            # kv_cache_dtype="fp8", # Optional: If using Ampere+ GPUs
        )
        print("vLLM Engine initialized successfully on Rank 0.")
        return inference_engine
    except Exception as e:
        print(f"ERROR initializing vLLM: {e}")
        print("vLLM initialization failed. Inference-related steps will be skipped.")
        return None 