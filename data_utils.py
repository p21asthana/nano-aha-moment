from typing import Any, Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer

# Import constants from config (assuming config.py is in the same directory)
from config import SYSTEM_MESSAGE, PROMPT_TEMPLATE


def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    system_message: str = SYSTEM_MESSAGE, # Use default from config
    prompt_template: str = PROMPT_TEMPLATE, # Use default from config
):
    """Preprocesses a single example from the dataset."""
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    # Tokenize and return input_ids and the prompt string itself
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False) # Important: Don't add generation prompt here
    prompt = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False)

    return {"prompt": prompt, "input_ids": input_ids}

def load_and_preprocess_data(tokenizer: AutoTokenizer, num_proc: int) -> tuple[dict, dict]:
    """Loads the dataset, preprocesses it, and splits into train/test."""
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.map(
        preprocess_example,
        num_proc=num_proc,
        fn_kwargs={
            "tokenizer": tokenizer,
            # Pass constants explicitly or rely on defaults in preprocess_example
            "system_message": SYSTEM_MESSAGE,
            "prompt_template": PROMPT_TEMPLATE,
        },
        remove_columns=dataset.column_names # Keep only prompt and input_ids
    )
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    dataset_info = {
        "train_len": len(train_dataset),
        "test_len": len(test_dataset)
    }
    print(f"Train dataset size: {dataset_info['train_len']}")
    print(f"Test dataset size: {dataset_info['test_len']}")

    return train_dataset, test_dataset, dataset_info 