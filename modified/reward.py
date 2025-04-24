import re
import numpy as np
from typing import List, Dict, Any, Tuple

def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        # Remove this line if <think> is always present in the actual generation
        # completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        # Adjusted regex to handle potential missing <think> if generation starts after it
        regex = r"^(?:<think>)?([^<]*(?:<(?!/?think>)[^<]*)*)?<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None:
             # Fallback check if only answer tag is present
             match_answer_only = re.search(r"<answer>([\s\S]*?)<\/answer>$", completion, re.DOTALL)
             if match_answer_only is None:
                 return 0.0 # Neither format matches
             else:
                 answer_content = match_answer_only.group(1).strip()
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

        # Check if answer content matches the allowed pattern
        if not re.match(allowed_pattern, answer_content):
            # If it doesn't match, reward is 0.5
            return 0.5
        else:
            # If both format and pattern are correct, reward is 1
            return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0

def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        # completion = "<think>" + completion # Let's assume the input completion includes everything after <think>
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0

def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    """Computes the overall reward and individual components."""
    nums = sample["nums"]
    target = sample["target"]

    # Note: The 'completion' passed here starts *after* the initial prompt including "<think>"
    # Adjust reward functions if they expect the full sequence including the prompt.
    # The current reward functions seem designed for the completion part only.

    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(completion=completion, nums=nums, target=target)

    # Combine rewards (example: simple sum, could be weighted)
    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics 