import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should return your GPU ID, e.g., 0
print(torch.cuda.get_device_name(0))  # Should return your GPU name, e.g., NVIDIA GeForce GTX 1650

torch.cuda.empty_cache()
