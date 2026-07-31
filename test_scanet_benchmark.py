# =====================================================================
# test_scanet_benchmark.py — SCANet IR-SR 標準化測試 (對標 IRSRMamba/GPSMamba)
# ---------------------------------------------------------------------
# 功能:
#   1. 載入 SCANet + 權重 (strict 檢查, 不允許 silent mismatch)
#   2. 對 result-A / result-C 逐張推論並存出 SR 影像
#   3. 以 BasicSR 官方協定計算 PSNR/SSIM/MSE:
#        crop_border = scale, test_y_channel = True (同 IRSRMamba/GPSMamba 報表)
#      並同場加映「純灰階」協定 (誠實數字, 兩者差 ~1.32 dB)
#   4. 參數量 / FLOPs (thop, 以測試集平均解析度) / 平均推論時間
#   5. 印出論文可直接貼的比較表 (tab 分隔, Excel 可貼)
#
# 用法:
#   python test_scanet_benchmark.py                       # x2, best_epoch.pth
#   python test_scanet_benchmark.py --scale 4 --weight best_x4.pth
#   python test_scanet_benchmark.py --no-save             # 只算指標不存圖
# =====================================================================
import argparse
import datetime
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from basicsr_metrics import calculate_mse, calculate_psnr, calculate_ssim
from models.scanet import SCANet

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------
def natural_key(name):
    m = re.findall(r'\d+', name)
    return (int(m[0]) if m else 0, name)


def load_gray(path):
    """讀成單通道灰階 uint8 (result-A/C 是灰階複製 3 通道, convert('L') 無損)."""
    return np.array(Image.open(path).convert('L'))


# ---------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------
@torch.no_grad()
def super_resolve(model, lr_u8, scale, device, pad_multiple=1):
    """lr_u8: HxW uint8 -> sr_u8: (H*s)x(W*s) uint8.
    輸入正規化 (x-0.5)/0.5 對齊訓練。SCANet-v8 (條紋掃描/FFT) 原生支援任意
    尺寸, 預設不 padding (實測 pad 會輕微降低邊緣品質); 舊版架構需要 8 的
    倍數時用 --pad 8."""
    h, w = lr_u8.shape
    x = torch.from_numpy(lr_u8).float().div(255.).sub(0.5).div(0.5)
    x = x.unsqueeze(0).unsqueeze(0).to(device)
    ph = (pad_multiple - h % pad_multiple) % pad_multiple
    pw = (pad_multiple - w % pad_multiple) % pad_multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    out = model(x)
    out = out[:, :, :h * scale, :w * scale]
    out = (out + 1) / 2.0                       # 反正規化回 [0,1]
    sr = out.squeeze().float().cpu().numpy()
    return (sr * 255.).clip(0, 255).round().astype(np.uint8)


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def evaluate_dataset(model, name, lr_dir, hr_dir, out_dir, scale, device,
                     save_images=True, pad_multiple=1):
    files = sorted(os.listdir(lr_dir), key=natural_key)
    if save_images:
        os.makedirs(out_dir, exist_ok=True)

    rec = {k: [] for k in ('psnr_y', 'ssim_y', 'mse_y', 'psnr_g', 'ssim_g')}
    t_total, n = 0.0, 0

    for fname in files:
        lr = load_gray(os.path.join(lr_dir, fname))
        hr_path = os.path.join(hr_dir, fname)
        if not os.path.exists(hr_path):
            print(f'  [warn] 缺 HR, 跳過: {fname}')
            continue
        hr = load_gray(hr_path)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        sr = super_resolve(model, lr, scale, device, pad_multiple)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_total += time.time() - t0
        n += 1

        # HR 尺寸若非 LR 的整數倍, 依 BasicSR 慣例對齊到共同區域
        h = min(sr.shape[0], hr.shape[0])
        w = min(sr.shape[1], hr.shape[1])
        sr_c, hr_c = sr[:h, :w], hr[:h, :w]

        if save_images:
            Image.fromarray(sr).save(os.path.join(out_dir, fname))

        # --- 協定 1: BasicSR / IRSRMamba 報表協定 (Y channel) ---
        # 灰階複製成 3 通道 -> Y = 16 + 219/255*gray, 對應官方 test_y_channel=true
        sr3 = np.repeat(sr_c[..., None], 3, axis=2)
        hr3 = np.repeat(hr_c[..., None], 3, axis=2)
        rec['psnr_y'].append(calculate_psnr(hr3, sr3, crop_border=scale, test_y_channel=True))
        rec['ssim_y'].append(calculate_ssim(hr3, sr3, crop_border=scale, test_y_channel=True))
        rec['mse_y'].append(calculate_mse(hr3, sr3, crop_border=scale, test_y_channel=True))

        # --- 協定 2: 純灰階 (誠實值, 建議論文標註) ---
        rec['psnr_g'].append(calculate_psnr(hr_c, sr_c, crop_border=scale, test_y_channel=False))
        rec['ssim_g'].append(calculate_ssim(hr_c, sr_c, crop_border=scale, test_y_channel=False))

    avg = {k: float(np.mean(v)) for k, v in rec.items()}
    avg['ms_per_img'] = t_total / max(n, 1) * 1000
    avg['n'] = n
    print(f'[{name}] n={n}  |  Y-channel: PSNR {avg["psnr_y"]:.4f}  SSIM {avg["ssim_y"]:.4f}  '
          f'MSE {avg["mse_y"]:.4f}  |  gray: PSNR {avg["psnr_g"]:.4f}  SSIM {avg["ssim_g"]:.4f}  '
          f'|  {avg["ms_per_img"]:.0f} ms/img')
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default=os.path.join(ROOT, 'best_epoch.pth'))
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--no-save', action='store_true', help='只算指標不存 SR 圖')
    parser.add_argument('--tag', default=None, help='輸出資料夾名 (預設用日期)')
    parser.add_argument('--pad', type=int, default=1,
                        help='輸入 pad 到此倍數 (v8 不需要; 舊版小波架構用 8)')
    args = parser.parse_args()

    # Windows 主控台 (cp950) 中文輸出保險
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | scale: x{args.scale} | weight: {args.weight}')

    # --- 模型 ---
    model = SCANet(1, args.channels, args.scale).to(device)
    sd = torch.load(args.weight, map_location=device, weights_only=True)
    sd = {k: v for k, v in sd.items()
          if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(sd, strict=True)     # 不允許 silent mismatch
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- FLOPs: 以各測試集常見 LR 尺寸計 (印出時標明尺寸) ---
    flops_g = None
    try:
        from thop import profile
        dummy = torch.randn(1, 1, 135, 180).to(device)   # result-A/C 最常見 LR 尺寸
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
    except Exception as e:
        print(f'[warn] thop FLOPs 計算失敗: {e}')

    # --- 測試集 ---
    tag = args.tag or datetime.datetime.now().strftime('%m%d_bench')
    suffix = '' if args.scale == 2 else f'_x{args.scale}'
    datasets = {}
    for name in ('result-A', 'result-C'):
        base = os.path.join(ROOT, 'data', 'results', name)
        lr_dir = os.path.join(base, f'LR{suffix}')
        hr_dir = os.path.join(base, f'HR{suffix}')
        if not (os.path.isdir(lr_dir) and os.path.isdir(hr_dir)):
            print(f'[warn] 找不到 {lr_dir} 或 {hr_dir}, 跳過 {name}')
            continue
        datasets[name] = evaluate_dataset(
            model, name, lr_dir, hr_dir,
            os.path.join(base, f'{tag}'), args.scale, device,
            save_images=not args.no_save, pad_multiple=args.pad)

    # --- 論文表格 (tab 分隔, 可直接貼 Excel) ---
    print('\n' + '=' * 78)
    print('論文比較表 (BasicSR 協定: crop_border=scale, Y-channel) — 複製下方貼 Excel')
    print('=' * 78)
    header = 'Scale\tMethod\t#Params(K)\tFLOPs(G)'
    for name in datasets:
        header += f'\t{name} PSNR\t{name} SSIM'
    print(header)
    flops_str = f'{flops_g:.2f}' if flops_g is not None else 'N/A'
    row = f'x{args.scale}\tSCANet (Ours)\t{num_params/1e3:.1f}\t{flops_str}'
    for name, avg in datasets.items():
        row += f'\t{avg["psnr_y"]:.4f}\t{avg["ssim_y"]:.4f}'
    print(row)

    print('\n誠實協定 (純灰階, 建議論文另欄或註明):')
    for name, avg in datasets.items():
        print(f'  {name}: PSNR {avg["psnr_g"]:.4f} / SSIM {avg["ssim_g"]:.4f}')
    print(f'\n#Params: {num_params/1e3:.1f}K ({num_params/1e6:.3f}M)')
    if flops_g:
        print(f'FLOPs @ 135x180 LR (x{args.scale}): {flops_g:.2f} G')


if __name__ == '__main__':
    main()
