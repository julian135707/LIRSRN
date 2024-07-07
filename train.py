import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
from models.LIRSRN import LIRSRN
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile, clever_format
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

batch_size = 16
epochs = 300
learning_rate = 0.0001
decay_rate = 0.5
decay_interval = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_interval = 1

model = LIRSRN(1, 64, 2).to(device)

print(f"Model's state_dict:")
for param_tensor in model.state_dict():
    print(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")

print('=====================================')
print(f'Batch size: {batch_size}')
print(f'Number of epochs: {epochs}')
print(f'Learning rate: {learning_rate}')
print(f'Device: {device}')
print('=====================================')

input_tensor = torch.randn(1, 1, 96, 96).to(device)
model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {model_parameters}")
macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
macs, params = clever_format([macs, params], "%.3f")
print(f"MACs (Multiply-Accumulate Operations): {macs}")
print(f"Parameters: {params}")
summary(model, input_size=(1, 96, 96))

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform

        self.lr_filenames = os.listdir(lr_dir)
        self.hr_filenames = os.listdir(hr_dir)

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, idx):
        lr_filename = self.lr_filenames[idx]
        hr_filename = self.hr_filenames[idx]

        lr_path = os.path.join(self.lr_dir, lr_filename)
        hr_path = os.path.join(self.hr_dir, hr_filename)

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MyDataset('./dataset/DIV2K/train/LR', './dataset/DIV2K/train/HR', transform=transform)
val_dataset = MyDataset('./dataset/DIV2K/val/LR', './dataset/DIV2K/val/HR', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)


scheduler = StepLR(optimizer, step_size=decay_interval, gamma=decay_rate)
writer = SummaryWriter(os.path.join("samples", "logs", './checkpoints/log/0521_month'))

best_score = 0.0
best_epoch = 0
losses = []
psnrs = []
ssims = []


criterion_mse = nn.MSELoss()
criterion_smoothl1 = nn.SmoothL1Loss()

for epoch in range(epochs):
    start_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    epoch_loss = 0.0


    for i, (lr_images, hr_images) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', ncols=100)):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad()


        batch_size, channels, height, width = lr_images.size()


        outputs = model(lr_images)


        target_height, target_width = hr_images.size()[2:]


        outputs = nn.functional.interpolate(outputs, size=(target_height, target_width), mode='bilinear',
                                            align_corners=True)

        loss_mse = criterion_mse(outputs, hr_images)

        loss_smoothl1 = criterion_smoothl1(outputs, hr_images)

        loss = loss_mse + loss_smoothl1
        writer.add_scalar('train/loss_smoothl1', loss_smoothl1, epoch)
        writer.add_scalar('train/loss_mse', loss_mse, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()


    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], Average Total Loss: {avg_loss:.4f}')

    if (epoch+1) % save_interval == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f'Saved model weights to model_epoch_{epoch+1}.pth')


    total_psnr = 0.0
    total_ssim = 0.0
    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    with torch.no_grad():
        for val_lr_images, val_hr_images in val_loader:
            val_lr_images = val_lr_images.to(device)
            val_hr_images = val_hr_images.to(device)

            val_outputs = model(val_lr_images)

            batch_size, channels, height, width = val_hr_images.size()

            val_outputs = nn.functional.interpolate(val_outputs, size=(height, width), mode='bilinear',
                                                    align_corners=False)

            val_outputs = val_outputs.detach().cpu().numpy()
            for i in range(val_outputs.shape[0]):
                val_hr_image = val_hr_images[i].detach().cpu().numpy()

                val_output_image = val_outputs[i].transpose(1, 2, 0) * 255

                val_hr_image = val_hr_images[i].cpu().numpy().transpose(1, 2, 0) * 255

                val_output_image = np.array(val_output_image) / 255.0
                val_hr_image = np.array(val_hr_image) / 255.0

                val_psnr = psnr(val_output_image, val_hr_image, data_range=1.0)
                val_ssim = ssim(val_output_image, val_hr_image, win_size=7, data_range=255, channel_axis=2)
                total_psnr += val_psnr
                total_ssim += val_ssim

    avg_psnr = total_psnr / len(val_dataset)
    avg_ssim = total_ssim / len(val_dataset)
    psnrs.append(avg_psnr)
    ssims.append(avg_ssim)
    writer.add_scalar('val/PSNR', avg_psnr, epoch)
    writer.add_scalar('val/SSIM', avg_ssim, epoch)
    print(f'Epoch [{epoch + 1}/{epochs}], Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}')

    score = avg_psnr
    score2 = avg_ssim
    if score > best_score:
        best_score = score
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f'best_epoch.pth')
        print(f'Saved model weights to model_epoch_{epoch + 1}.pth')
    print(f'Best Score: {best_score:.4f} at Epoch {best_epoch}')

    epoch_time = time.time() - start_time

    print(f'Epoch completed in {epoch_time:.2f} seconds')
    if (epoch + 1) > 200:
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        print(f'Saved model weights to model_epoch_{epoch + 1}.pth')
    scheduler.step()

    current_progress = (epoch + 1) / epochs * 100
    remaining_time = epoch_time * (epochs - epoch - 1)
    print(f'Progress: {current_progress:.2f}%  Remaining time: {remaining_time:.2f} seconds')

plt.figure()
plt.plot(range(1, epochs + 1), losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./graphs/training_loss.png')
plt.show()

plt.figure()
plt.plot(range(1, epochs + 1), psnrs, label='PSNR')
plt.title('Peak Signal to Noise Ratio (PSNR)')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()
plt.savefig('./graphs/psnr.png')
plt.show()

plt.figure()
plt.plot(range(1, epochs + 1), ssims, label='SSIM')
plt.title('Structural Similarity Index Measure (SSIM)')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()
plt.savefig('./graphs/ssim.png')
plt.show()

torch.save(model.state_dict(), f'best_model_epoch_{best_epoch}.pth')
print(f'Saved best model weights to best_model_epoch_{best_epoch}.pth')
