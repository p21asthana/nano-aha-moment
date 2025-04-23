# Unset LD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# Launch with deepspeed (adjust args as needed)
deepspeed --num_gpus=4 nano_r1_script.py --model_name Qwen/Qwen2.5-3B
