export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"


CUDA_VISIBLE_DEVICES=0 python models/flux/src/sample.py --prompt_file resources/prompts/prompt.txt \
  --width 1024 --height 1024 \
  --model_name flux-dev \
  --add_sampling_metadata \
  --output_dir /root/autodl-tmp/TaylorSeer/Hermite-FLUX/results/taylor/interval-6\
  --num_steps 50 \
  --add_sampling_metadata
