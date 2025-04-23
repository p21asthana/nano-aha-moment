# conda activate /home/ec2-user/SageMaker/envs/nano-aha

# Unset LD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# Launch with torchrun on g5.12xlarge (4 GPUs) using Policy Model fallback for generation
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 \
  main.py                                  \
  --model_name Qwen/Qwen2.5-3B             \
  --episodes_per_iteration 16              \
  --per_device_batch_size 2                \
  --num_iterations 50                      \
  --no_vllm # Use the fallback generation path