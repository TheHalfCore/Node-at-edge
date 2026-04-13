import torch
torch.cuda.empty_cache()

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"