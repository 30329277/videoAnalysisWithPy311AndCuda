import torch

print("CUDA Available:", torch.cuda.is_available())
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
