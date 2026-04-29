# BRACE: Taming Sharp Irregularities via Barycentric Rational Forecasting for Fast Diffusion Transformers Inference


## abstract 
Diffusion Transformers (DiTs) have demonstrated exceptional performance in high-fidelity image and video generation. To alleviate their massive computational overhead, temporal feature caching has been proposed to bypass redundant computations. However, existing cache-then-forecast methods driven by derivative-based polynomials often cause severe quality degradation under high acceleration due to unstable long-step predictions. To address this bottleneck, we propose Barycentric Rational Forecasting with Chebyshev Enhancement (BRACE). Motivated by the observation that DiT feature trajectories are globally smooth yet frequently exhibit sharp irregularities and local non-smoothness, BRACE shifts the paradigm from derivative-driven polynomial extrapolation to feature-driven rational forecasting. Specifically, it maintains a local sliding window to cache sparse historical features and leverages adapted Chebyshev weights to formulate a barycentric rational function, directly aggregating these raw features to ensure numerical stability. Extensive experiments demonstrate that BRACE achieves state-of-the-art quality efficiency trade-offs across various DiT architectures with negligible computational overhead.

## Models & Quick Start
Currently, this repository supports accelerating the following models.

💡 Codebase Note: In the execution scripts and source code, the BRACE method is abbreviated as bary (Barycentric). Please use --method bary when running the inference scripts.

### BRACE-DiT-XL-2
#### 1. Prepare Environment

```bash
cd BRACE-DiT
conda env create -f environment.yml
conda activate DiT
pip install flash-attention
```
#### 2. Download Checkpoints

Simply follow the official documentation to download the necessary checkpoints.
#### Distributed Data Parallel (DDP) Inference
```bash
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 4 \
  --max-order 4 \
  --num-fid-samples 50000
```

### BRACE-FLUX
#### 1. Set Up Conda Environment

Follow the official documentation instructions to create the Conda environment:

```bash
conda create -n flux python=3.10
conda activate flux
pip install -e ".[all]"
```

#### 2. Download Checkpoints with Your Hugging Face Token

If you experience connection issues with Hugging Face, you can use the Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Make sure you have obtained the necessary permissions and exported your token:

```bash
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-dev --local-dir /path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-schnell --local-dir /path/to/save/pretrained_models/black-forest-labs/FLUX.1-schnell
huggingface-cli download --token YOUR_HF_TOKEN --resume-download google/t5-v1_1-xxl --local-dir /path/to/save/pretrained_models/google/t5-v1_1-xxl
huggingface-cli download --token YOUR_HF_TOKEN --resume-download openai/clip-vit-large-patch14 --local-dir /path/to/save/pretrained_models/openai/clip-vit-large-patch14
```

<details>
  <summary>Download Checkpoints on AutoDL</summary>
  
  ```bash
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-dev --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-schnell --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-schnell
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download google/t5-v1_1-xxl --local-dir /root/autodl-tmp/pretrained_models/google/t5-v1_1-xxl
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download openai/clip-vit-large-patch14 --local-dir /root/autodl-tmp/pretrained_models/openai/clip-vit-large-patch14
  ```
</details>

#### 3. Set Environment Variables (in `.bashrc` file)

```bash
export FLUX_SCHNELL="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export FLUX_DEV="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"
```

<details>
  <summary>Set Environment Variables for AutoDL</summary>
  
  ```bash
  export FLUX_SCHNELL="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
  export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
  export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"
  ```
</details>

#### 4. Sampling with BRACE

```bash
python src/sample.py --prompt_file </path/to/your/prompt.txt> \
  --width 1024 --height 1024 --model_name flux-dev \
  --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> --num_steps 50
```


#### 5. Modify Configuration for Custom Needs

The current framework supports testing multiple methods. You can add your own methods or modify existing ones in:
`BRACE-FLUX/src/flux/modules/cache_functions/cache_init.py`


### BRACE-HunYuan Video

#### **1. Prepare Environment**

Follow the official **HunyuanVideo** documentation to set up the environment.

<details>
  <summary><strong>Conda Environment Setup</strong></summary>

  ```bash
  # 1. Create the Conda environment
  conda create -n HunyuanVideo python==3.10.9

  # 2. Activate the environment
  conda activate HunyuanVideo

  # 3. Install PyTorch and dependencies
  # For CUDA 11.8
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  # For CUDA 12.4
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

  # 4. Install required Python dependencies
  python -m pip install -r requirements.txt

  # 5. Install FlashAttention v2 for acceleration (requires CUDA 11.8 or later)
  python -m pip install ninja
  python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

  # 6. Install xDiT for parallel inference (recommended with PyTorch 2.4.0 and FlashAttention 2.6.3)
  python -m pip install xfuser==0.4.0
  ```

  If you encounter a **floating point exception (core dump)** on specific GPUs, try the following solutions:

  ```bash
  # Option 1: Ensure CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 are installed
  # (Alternatively, use our prebuilt CUDA 12 Docker image)
  pip install nvidia-cublas-cu12==12.4.5.8
  export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

  # Option 2: Force using the CUDA 11.8-compiled version of PyTorch and dependencies
  pip uninstall -r requirements.txt  # Uninstall all packages
  pip uninstall -y xfuser
  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  pip install ninja
  pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
  pip install xfuser==0.4.0
  ```

</details>

#### **2. Download Checkpoints**

Refer to the [**checkpoint download guide**](TaylorSeer-HunyuanVideo/ckpts/README.md)
---
#### Run Samples

##### **Single Video Inference**

Run inference on a single video. Feel free to adjust the parameters and prompt as needed.

```bash
cd HunyuanVideo
python3 sample_video.py \
  --video-size 480 640 \
  --video-length 65 \
  --infer-steps 50 \
  --seed 42 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse \
  --use-cpu-offload \
  --save-path /path/to/save/videos
```

---

##### **Multi-Video Inference (VBench Testing)**

We provide a script to test **VBench** on HunyuanVideo, supporting multi-GPU parallel inference.
```bash
cd HunyuanVideo
# Run VBench evaluation:
# ./sample_vbench.sh <full_info_path> <Num_Devices> <SEED> <Num_Samples> <Video_Save_Path> <Path2Log>
./eval/sample_vbench.sh ./eval/ 1 42 5 /path/to/save/vbench/videos /path/to/save/logger/files
```

### 👍 Acknowledgements
- Thanks to DiT for their great work and codebase upon which we build BRACE-DiT.
- Thanks to FLUX for their great work and codebase upon which we build BRACE-FLUX.
- Thanks to HunyuanVideo for their great work and codebase upon which we build BRACE-HunyuanVideo.
- Thanks to TaylorSeer for their great work and codebase