import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import re
from thop import profile
from models.LIRSRN import LIRSRN

batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir), key=lambda x: int(re.findall(r'\d+', x)[0]))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = MyDataset('./dataset/Testing Dataset/result-A/LR', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LIRSRN(1, 64, 2).to(device)
state_dict = torch.load('./checkpoints/weight.pth', map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()

output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i, images in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        outputs = (outputs + 1) / 2.0

        output_images = outputs.cpu().numpy()
        for j in range(output_images.shape[0]):
            output_image = output_images[j][0]
            output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)
            output_path = os.path.join(output_dir, f'fused{i + 1}.png')
            output_image.save(output_path)
            print(f'Saved output image: {output_path}')

num_params = sum(p.numel() for p in model.parameters()) / 1e3
print(f"Number of parameters: {num_params:.2f}K")
input_shape = (batch_size, 1, 96, 96)
flops, params = profile(model, inputs=(torch.randn(input_shape).to(device),))
gflops = flops / 1e9
print(f"GFLOPs: {gflops:.2f}G")
