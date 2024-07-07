import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(output, target):
    output = output.clamp(0, 1)
    mse = torch.mean((output - target) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def calculate_ssim(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    ssim = structural_similarity(output, target, win_size=7, data_range=target.max() - target.min(), multichannel=True)
    return ssim

def resize(image, size, anti_aliasing=True):
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
        resized_image = resize(image, size, anti_aliasing=anti_aliasing)
        resized_image = np.transpose(resized_image, (2, 0, 1))
        return resized_image
    else:
        from skimage.transform import resize as sk_resize
        return sk_resize(image, size, anti_aliasing=anti_aliasing)

