export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"

python models/flux/src/sample.py --prompt_file /root/autodl-tmp/TaylorSeer/TaylorSeer-FLUX/prompt_test.txt \
  --width 1024 --height 1024 --model_name flux-dev \
  --add_sampling_metadata --output_dir /root/autodl-tmp/TaylorSeer/TaylorSeer-FLUX/flops_test \
  --num_steps 50 --test_FLOPs
