# Unset LD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# Launch with deepspeed (adjust args as needed)
deepspeed --num_gpus=4 main.py --model_name Qwen/Qwen2.5-3B --episodes_per_iteration=16 --per_device_batch_size=1 # Add other args as needed