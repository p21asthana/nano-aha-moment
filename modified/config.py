from pathlib import Path

# --- Paths ---
SCRATCH = Path.home() / "scratch"
HF_HOME = SCRATCH / "hf_home"
DEFAULT_EXP_DIR_BASE = SCRATCH / "deepseek_hackathon"

# --- Default Model Names ---
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B"
DEFAULT_MODEL_CHAT_NAME_SUFFIX = "-Instruct"

# --- Default RL Parameters ---
DEFAULT_KL_COEFF = 0.001
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NUM_ITERATIONS = 1000
DEFAULT_EPISODES_PER_ITERATION = 64 # Global batch size
DEFAULT_GENERATIONS_PER_SAMPLE = 4

# --- Default Training Hyperparameters ---
DEFAULT_PER_DEVICE_BATCH_SIZE = 4 # Micro batch size per GPU
DEFAULT_LEARNING_RATE = 1e-6

# --- Default Sampling Parameters ---
DEFAULT_MAX_RESPONSE_TOKENS = 1024
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = -1  # -1 means disabled

# --- Prompts ---
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)
PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)

# --- Default DeepSpeed Config Snippets (can be merged/overridden in main script) ---
# These are partial configs, the full ones are built dynamically based on args/world_size
DEFAULT_OPTIMIZER_PARAMS = {
    "type": "AdamW",
    "params": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    },
}

DEFAULT_SCHEDULER_PARAMS = {
    "type": "WarmupLR",
    "params": {
        "warmup_min_lr": 0,
        "warmup_num_steps": 100
    }
}

DEFAULT_ZERO_STAGE_3_CONFIG = {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": True},
    "offload_param": {"device": "cpu", "pin_memory": True},
    "overlap_comm": True,
    "contiguous_gradients": True,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "sub_group_size": 1e9
}

DEFAULT_ZERO_REF_STAGE_3_CONFIG = {
    "stage": 3,
    "offload_param": {"device": "cpu", "pin_memory": True},
    "contiguous_gradients": True,
    "reduce_bucket_size": 5e7,
} 