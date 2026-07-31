# =====================================================================
# basicsr_metrics.py — BasicSR-compatible PSNR / SSIM (numpy, uint8 [0,255])
# ---------------------------------------------------------------------
# 完全對齊 BasicSR (XPixelGroup) 的官方實作與慣例, 即 IRSRMamba / GPSMamba /
# MambaIR / SwinIR 等論文報表所用的指標計算方式:
#   * crop_border = scale (x2 裁 2 px, x4 裁 4 px)
#   * test_y_channel=True: BGR -> Y (BT.601, 值域 [16,235]) 後計算
#   * SSIM: 11x11 Gaussian (sigma=1.5), C1=(0.01*255)^2, C2=(0.03*255)^2
# 注意: 灰階影像複製成 3 通道後轉 Y 等於線性壓縮 (Y = 16 + 219/255 * gray),
# 會使 PSNR 比直接灰階計算高 ~1.32 dB。所以「跟別人比」時必須用同一種協定;
# 本模組兩種都提供, 論文報表建議標明使用哪一種。
# =====================================================================
import cv2
import numpy as np


def reorder_image(img, input_order='HWC'):
    if input_order not in ('HWC', 'CHW'):
        raise ValueError(f'Wrong input_order {input_order}')
    if img.ndim == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def bgr2ycbcr(img, y_only=False):
    """img: float32 [0,1] 或 uint8 [0,255], HWC BGR. 回傳同型別."""
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.uint8:
        img /= 255.
    if y_only:
        out = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out = np.matmul(img, [[24.966, 112.0, -18.214],
                              [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) + [16, 128, 128]
    if img_type == np.uint8:
        out = out.round()
    else:
        out /= 255.
    return out.astype(img_type)


def to_y_channel(img):
    """uint8/float HWC. 3 通道 -> BT.601 Y ([16,235] 尺度); 灰階直接原樣回傳."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    assert img1.shape == img2.shape, f'shapes differ: {img1.shape} vs {img2.shape}'
    img1 = reorder_image(img1, input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order).astype(np.float64)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def calculate_mse(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    assert img1.shape == img2.shape, f'shapes differ: {img1.shape} vs {img2.shape}'
    img1 = reorder_image(img1, input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order).astype(np.float64)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    return np.mean((img1 - img2) ** 2)


def _ssim_single(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    assert img1.shape == img2.shape, f'shapes differ: {img1.shape} vs {img2.shape}'
    img1 = reorder_image(img1, input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order).astype(np.float64)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    ssims = [_ssim_single(img1[..., i], img2[..., i]) for i in range(img1.shape[2])]
    return np.array(ssims).mean()
