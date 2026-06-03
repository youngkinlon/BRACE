from typing import Any, Dict, Optional, Tuple, Union
import time
from diffusers import DiffusionPipeline
from diffusers.pipelines.flux import FluxPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
from forwards import (taylorseer_flux_single_block_forward, 
                        taylorseer_flux_double_block_forward, 
                        taylorseer_flux_forward)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42
prompt = "An image of a squirrel in Picasso style"
pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
#pipeline = DiffusionPipeline.from_pretrained("/root/autodl-tmp/black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
#pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TaylorSeer settings
pipeline.transformer.__class__.num_steps = num_inference_steps

pipeline.transformer.__class__.forward = taylorseer_flux_forward

for double_transformer_block in pipeline.transformer.transformer_blocks:
    double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
    
for single_transformer_block in pipeline.transformer.single_transformer_blocks:
    single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

pipeline.to("cuda")

parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")

torch.cuda.reset_peak_memory_stats()
#start_time = time.time()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
img = pipeline(
    prompt, 
    num_inference_steps=num_inference_steps,
    generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end) * 1e-3
peak_memory = torch.cuda.max_memory_allocated(device="cuda")

img.save("{}.png".format('TaylorSeer_' + prompt))

print(
    f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
)