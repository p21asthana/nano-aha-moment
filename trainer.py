import os
import gc
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel, AutoModelForCausalLM # Added AutoModelForCausalLM
from vllm import LLM, SamplingParams # Added LLM
from tqdm import trange
import wandb
from typing import Any, Dict, List, Tuple, Union

# Import from project modules
from reward import compute_reward
from utils import (
    compute_token_log_probs,
    prepare_model_inputs,
    evaluate_on_test_set,
    dump_episodes,
    load_model_into_vllm, # Needed for update within loop
)


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    tokenizer: AutoTokenizer,
    eos_token_id: int,
    eos_token: str,
    generations_per_sample: int,
    local_rank: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.
    (Copied from original script, requires reward.compute_reward)
    """
    # Only rank 0 should perform this computation
    if local_rank != 0:
        return {}, {}

    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * generations_per_sample

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + generations_per_sample))
        for i in range(0, len(all_generations), generations_per_sample)
    ]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward(resp, sample, eos_token) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * generations_per_sample)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats

def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    temperature: float,
    kl_coefficient: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.
    (Copied from original script, requires utils.compute_token_log_probs)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    advantages = batch["advantages"]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()
    local_total_response_len = labels_mask.sum()

    global_total_response_len_tensor = torch.tensor(
        [local_total_response_len], device=local_total_response_len.device
    )
    if dist.is_initialized():
        dist.all_reduce(global_total_response_len_tensor, op=dist.ReduceOp.SUM)
    global_total_response_len = global_total_response_len_tensor.item()

    if global_total_response_len == 0:
        global_total_response_len = 1

    with torch.no_grad():
        ref_logps = compute_token_log_probs(reference_model, model_inputs, temperature)

    logps = compute_token_log_probs(policy_model, model_inputs, temperature)

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask

    entropy = (-logps * labels_mask).sum() / max(1, local_total_response_len)

    policy_loss = -logps * advantages[..., 1:]
    policy_loss = policy_loss * labels_mask

    loss = (policy_loss + kl_coefficient * kl_penalty).sum() / global_total_response_len

    metrics = {
        "policy_loss": (policy_loss.sum() / global_total_response_len).item(),
        "kl_penalty": (kl_penalty.sum() / global_total_response_len).item(),
        "entropy": entropy.item(),
        "local_response_len": local_total_response_len.item(),
        "global_response_len": global_total_response_len,
    }

    return loss, metrics

def train_loop(
    args: argparse.Namespace,
    config: dict, # Pass loaded config/constants
    policy_model: DeepSpeedEngine,
    reference_model: DeepSpeedEngine,
    inference_engine: LLM,
    tokenizer: AutoTokenizer,
    train_dataset: Any, # Type hint properly later
    test_dataset: Any,
    dataset_info: Dict[str, int],
    begin_iter: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
):
    """Main training loop."""
    is_rank_0 = local_rank == 0
    EXP_DIR = config["EXP_DIR"]
    EOS_TOKEN = config["EOS_TOKEN"]
    EOS_TOKEN_ID = config["EOS_TOKEN_ID"]
    GENERATIONS_PER_SAMPLE = config["GENERATIONS_PER_SAMPLE"]
    EPISODES_PER_ITERATION = config["EPISODES_PER_ITERATION"]
    PER_DEVICE_BATCH_SIZE = config["PER_DEVICE_BATCH_SIZE"]
    NUM_ITERATIONS = config["NUM_ITERATIONS"]
    TEMPERATURE = args.temperature
    TOP_P = config["TOP_P"]
    TOP_K = config["TOP_K"]
    MAX_RESPONSE_TOKENS = config["MAX_RESPONSE_TOKENS"]
    KL_COEFFICIENT = args.kl_coeff

    for iteration in range(begin_iter, NUM_ITERATIONS):
        iter_start_time = time.time()
        if is_rank_0:
            print(f"\n--- Iteration {iteration}/{NUM_ITERATIONS} ---")

        metrics = {} # Metrics collected only on Rank 0

        # --- Evaluation (Rank 0 Only) ---
        eval_stats = None
        eval_episodes = None
        if iteration % 25 == 0 and is_rank_0 and inference_engine is not None:
            print("Rank 0: Evaluating on eval set...")
            eval_start_time = time.time()
            try:
                eval_episodes, eval_stats = evaluate_on_test_set(
                    inference_engine=inference_engine,
                    test_dataset=test_dataset,
                    tokenizer=tokenizer,
                    eos_token=EOS_TOKEN,
                    eval_sampling_params=SamplingParams(
                        temperature=0.3,
                        max_tokens=1024,
                        n=1,
                        detokenize=False,
                        stop_token_ids=[EOS_TOKEN_ID] if EOS_TOKEN_ID is not None else [],
                    ),
                    # Need to pass compute_reward itself or wrap it
                    reward_func=lambda completion, sample: compute_reward(completion, sample, EOS_TOKEN),
                )
                eval_episode_table = dump_episodes(
                    episodes=eval_episodes,
                    episodes_stats=eval_stats,
                    exp_dir=EXP_DIR,
                    tokenizer=tokenizer,
                    iteration=iteration,
                    is_eval=True,
                    local_rank=local_rank,
                )
                wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})
            except Exception as e:
                 print(f"ERROR during evaluation: {e}")
                 eval_stats = None
            print(f"Rank 0: Evaluation took {time.time() - eval_start_time:.2f} seconds")

        # --- Generate Episodes (Rank 0 Only) ---
        episodes = {}
        episodes_stats = {}
        generation_data = None
        gen_time = 0

        if is_rank_0:
            if inference_engine is None:
                print("Rank 0: Skipping episode generation as vLLM engine failed to initialize.")
                generation_data = {"episodes": {}, "episodes_stats": {}}
            else:
                gen_start_time = time.time()
                num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
                train_len = dataset_info['train_len']
                if train_len > 0 and num_samples > 0:
                     size = min(num_samples, train_len) if not (len(train_dataset) < num_samples) else num_samples
                     indices = np.random.choice(train_len, size=size, replace=len(train_dataset) < num_samples)
                     samples = train_dataset.select(indices).to_list()
                else:
                     samples = []

                if samples:
                    print(f"Rank 0: Generating {len(samples) * GENERATIONS_PER_SAMPLE} responses...")
                    try:
                        outputs = inference_engine.generate(
                            prompt_token_ids=[s['input_ids'] for s in samples],
                            sampling_params=SamplingParams(
                                n=GENERATIONS_PER_SAMPLE,
                                temperature=TEMPERATURE,
                                top_p=TOP_P,
                                top_k=TOP_K,
                                max_tokens=MAX_RESPONSE_TOKENS,
                                detokenize=False,
                                stop_token_ids=[EOS_TOKEN_ID] if EOS_TOKEN_ID is not None else [],
                            ),
                            use_tqdm=False
                        )
                        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
                        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]

                        print(f"Rank 0: Generated {len(all_generations)} responses")
                        gc.collect()
                        torch.cuda.empty_cache()

                        gen_time = time.time() - gen_start_time
                        print(f"Rank 0: Generation took {gen_time:.2f} seconds")

                        print("Rank 0: Creating training episodes...")
                        episodes, episodes_stats = create_training_episodes(
                            samples,
                            all_generations,
                            all_finish_reasons,
                            tokenizer,
                            EOS_TOKEN_ID,
                            EOS_TOKEN,
                            GENERATIONS_PER_SAMPLE,
                            local_rank=local_rank,
                        )

                        episode_table = dump_episodes(
                            episodes=episodes,
                            episodes_stats=episodes_stats,
                            exp_dir=EXP_DIR,
                            tokenizer=tokenizer,
                            iteration=iteration,
                            local_rank=local_rank,
                        )
                        generation_data = {"episodes": episodes, "episodes_stats": episodes_stats, "episode_table": episode_table, "gen_time": gen_time}

                    except Exception as e:
                        print(f"ERROR during generation or episode creation: {e}")
                        generation_data = {"episodes": {}, "episodes_stats": {}}

                else:
                     print("Rank 0: No samples to generate/train on.")
                     generation_data = {"episodes": {}, "episodes_stats": {}}
        else:
             generation_data = None

        # --- Broadcast Generation Data --- #
        if world_size > 1:
            broadcast_list = [generation_data]
            dist.broadcast_object_list(broadcast_list, src=0)
            if not is_rank_0:
                generation_data = broadcast_list[0]
                # print(f"Rank {local_rank} received generation data from Rank 0.") # Less verbose

        # --- Extract Data on All Ranks --- #
        episodes = generation_data.get("episodes", {})
        episodes_stats = generation_data.get("episodes_stats", {})
        if is_rank_0:
            metrics.update({k: v for k, v in episodes_stats.items() if v})
            gen_time = generation_data.get("gen_time", 0)
            if gen_time > 0:
                metrics["generation_time"] = [gen_time]
            episode_table = generation_data.get("episode_table")

        # --- Skip Training if No Episodes --- #
        if not episodes or not episodes.get("all_query_token_ids"):
            if is_rank_0:
                print("Skipping training step as no episodes were generated.")
            if world_size > 1:
                dist.barrier()
            continue

        # --- Training Step (All Ranks) --- #
        train_start_time = time.time()
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=device,
        )

        policy_model.train()
        reference_model.eval()

        local_metrics = {
            "loss": [], "policy_loss": [], "kl_penalty": [], "entropy": [], "grad_norm": []
        }

        # *** DeepSpeed Training Batch Logic - Needs Refinement ***
        # DeepSpeed usually integrates with a DataLoader.
        # Passing the full `model_inputs` to the engine and letting it handle
        # micro-batching might be the intended way, or defining a custom DataLoader.
        # For now, we perform a single backward/step assuming the engine handles accumulation.
        # This simplified loop might need adjustment based on DeepSpeed's API for manual data feeding.

        # Perform one training step - DeepSpeed handles accumulation
        loss, loss_metrics = compute_pg_loss(
                # For ZeRO-3, pass the engine directly? Or module? Check docs. Assume engine for now.
                policy_model=policy_model,
                reference_model=reference_model,
                batch=model_inputs, # Pass full batch, assume engine handles slicing/accumulation
                temperature=TEMPERATURE,
                kl_coefficient=KL_COEFFICIENT,
            )
        policy_model.backward(loss)
        grad_norm = policy_model.get_global_grad_norm()
        policy_model.step()

        # Log metrics from this step (representing accumulated gradients)
        local_metrics["loss"].append(loss.item())
        local_metrics["grad_norm"].append(grad_norm.item() if grad_norm is not None else 0.0)
        for k, v in loss_metrics.items():
            if isinstance(v, (int, float)):
                local_metrics.setdefault(k, []).append(v)
        # *** End of Simplified Training Step Logic ***

        train_time = time.time() - train_start_time
        if is_rank_0:
            print(f"Rank 0: Training step took {train_time:.2f} seconds")

        # --- Aggregate Metrics --- #
        if world_size > 1:
            gathered_metrics_list = [None] * world_size
            dist.gather_object(local_metrics, gathered_metrics_list if is_rank_0 else None, dst=0)
            if is_rank_0:
                aggregated_metrics = {}
                metrics.pop("episode_table", None)
                aggregated_metrics.update({f"train/{k}": np.mean(v) for k, v in metrics.items() if isinstance(v, list) and v})
                keys_to_aggregate = ["loss", "policy_loss", "kl_penalty", "entropy", "grad_norm"]
                for k in keys_to_aggregate:
                    all_values = [m[k][0] for m in gathered_metrics_list if m and k in m and m[k]] # Get the single value from each rank
                    if all_values:
                        aggregated_metrics[f"train/{k}"] = np.mean([v for v in all_values if v is not None])
                    else:
                        aggregated_metrics[f"train/{k}"] = 0.0
                metrics = aggregated_metrics
        else:
             metrics.pop("episode_table", None)
             train_metrics = {f"train/{k}": np.mean(v) for k, v in local_metrics.items() if v}
             metrics.update(train_metrics)

        # --- Update vLLM Weights (Rank 0 Only - Best Effort) --- #
        if is_rank_0 and inference_engine is not None:
            # print("Rank 0: Updating vLLM weights (Best effort with ZeRO-3)...")
            update_start_time = time.time()
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # TODO: Implement reliable weight loading from ZeRO-3 if needed
                # Currently skipping as it's complex and memory-intensive
                # load_model_into_vllm(policy_model, inference_engine)
                if iteration % 5 == 0: # Print warning less often
                     print("WARNING: Skipping vLLM weight update due to ZeRO-3 complexity.")
                pass
            except Exception as e:
                print(f"ERROR updating vLLM weights: {e}")
            # print(f"Rank 0: vLLM weight update attempt took {time.time() - update_start_time:.2f} seconds")

        # --- Log Metrics (Rank 0 Only) --- #
        if is_rank_0:
            print("Rank 0: Logging metrics...")
            try:
                current_lr = policy_model.get_lr()[0]
            except: # Fallback if get_lr not available
                 # Try accessing optimizer directly (might be wrapped differently)
                 try:
                     current_lr = policy_model.optimizer.param_groups[0]['lr']
                 except:
                     current_lr = args.learning_rate # Fallback to arg
            metrics["train/learning_rate"] = current_lr

            logs = {
                "iteration": iteration,
                f"episodes/iter_{iteration:06d}": episode_table if 'episode_table' in locals() and episode_table is not None else wandb.Table(columns=[]),
                **metrics,
            }
            if eval_stats is not None:
                valid_eval_stats = {f"eval/{k}": np.mean(v) for k, v in eval_stats.items() if isinstance(v, list) and v}
                logs.update(valid_eval_stats)

            if wandb.run is not None:
                 wandb.log(logs)
            else:
                 print("Wandb not initialized, skipping logging.")

            selected_keys = [
                "train/loss", "train/kl_penalty", "train/rewards",
                "train/reward_metrics/format_reward", "train/reward_metrics/equation_reward",
                "eval/rewards", "eval/reward_metrics/format_reward", "eval/reward_metrics/equation_reward",
                "train/grad_norm",
            ]
            selected_metrics = {k: f"{logs[k]:.4f}" for k in selected_keys if k in logs and isinstance(logs[k], (int, float, np.number))}
            print(f"Iteration {iteration} Key Metrics: {selected_metrics}")
            iter_time = time.time() - iter_start_time
            print(f"Iteration {iteration} Total Time: {iter_time:.2f} seconds")

        # --- Save Checkpoint (All Ranks) --- #
        if iteration % 50 == 0 and iteration != 0:
             save_tag = f"ckpt_{iteration:06d}"
             checkpoint_dir = EXP_DIR / "checkpoints"
             if is_rank_0:
                 print(f"Initiating checkpoint saving for iteration {iteration} with tag '{save_tag}' to {checkpoint_dir}...")

             # All ranks must call save_checkpoint for ZeRO-3
             policy_model.save_checkpoint(str(checkpoint_dir), tag=save_tag)

             if is_rank_0:
                 print(f"Rank 0 finished initiating checkpoint save for tag '{save_tag}'. Check all ranks for completion.")

        # --- Barrier --- #
        if world_size > 1:
            dist.barrier()

    # --- End of Loop --- #
    if is_rank_0:
        if wandb.run is not None:
            wandb.finish()
        print("Training finished.") 