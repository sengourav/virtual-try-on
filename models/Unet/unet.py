import zipfile
import os
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

dataset_path = "/content/drive/MyDrive/virtual-try-on/dataset.zip"  # Uploaded file name
extract_dir = "viton_dataset"

# Unzip dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Check extracted files
print("Dataset Structure:")
for folder in os.listdir(extract_dir):
    print("-", folder)

"""# TrainDataLoader"""

class VITONTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(os.path.join(root_dir, 'train_img')) if f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]                     # e.g., 019310_0.jpg
        base_name = img_name.replace('_0.jpg', '')           # e.g., 019310

        # Build other file paths
        img_path = os.path.join(self.root_dir, 'train_img', f"{base_name}_0.jpg")
        cloth_path = os.path.join(self.root_dir, 'train_color', f"{base_name}_1.jpg")
        label_path = os.path.join(self.root_dir, 'train_label', f"{base_name}_0.png")

        # Load images
        img = Image.open(img_path).resize((192, 256))
        cloth = Image.open(cloth_path).resize((192, 256))
        label = Image.open(label_path).resize((192, 256), resample=Image.NEAREST)

        # Create agnostic person image (remove upclothes → label 4)
        img_np = np.array(img)
        label_np = np.array(label)
        agnostic_np = img_np.copy()
        agnostic_np[label_np == 4] = [128, 128, 128]

        # Apply transforms
        to_tensor = self.transform if self.transform else transforms.ToTensor()
        agnostic_tensor = to_tensor(Image.fromarray(agnostic_np))
        cloth_tensor = to_tensor(cloth)
        target_tensor = to_tensor(img)

        return agnostic_tensor, cloth_tensor, target_tensor


transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = VITONTrainDataset(root_dir='viton_dataset/ACGPN_TrainData', transform=transform)

class VITONTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(os.path.join(root_dir, 'test_img')) if f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]                   # e.g., 006203_0.jpg
        base_name = img_name.replace('_0.jpg', '')         # e.g., 006203

        img_path = os.path.join(self.root_dir, 'test_img', f"{base_name}_0.jpg")
        cloth_path = os.path.join(self.root_dir, 'test_color', f"{base_name}_1.jpg")
        label_path = os.path.join(self.root_dir, 'test_label', f"{base_name}_0.png")

        # Load images
        img = Image.open(img_path).resize((192, 256))
        cloth = Image.open(cloth_path).resize((192, 256))
        label = Image.open(label_path).resize((192, 256), resample=Image.NEAREST)

        # Agnostic image creation (remove upclothes)
        img_np = np.array(img)
        label_np = np.array(label)
        agnostic_np = img_np.copy()
        agnostic_np[label_np == 4] = [128, 128, 128]

        to_tensor = self.transform if self.transform else transforms.ToTensor()
        agnostic_tensor = to_tensor(Image.fromarray(agnostic_np))
        cloth_tensor = to_tensor(cloth)
        target_tensor = to_tensor(img)

        return agnostic_tensor, cloth_tensor, target_tensor

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(UNetGenerator, self).__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.down1 = block(in_channels, 64)
        self.down2 = block(64, 128)
        self.down3 = block(128, 256)
        self.down4 = block(256, 512)

        self.up1 = up_block(512, 256)
        self.up2 = up_block(512, 128)
        self.up3 = up_block(256, 64)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        return u4

from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=8,      # Adjust depending on your Colab GPU
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetGenerator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.L1Loss()

import time

num_epochs = 10  # You can increase this later

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for i, (agnostic, cloth, target) in enumerate(train_loader):
        agnostic = agnostic.to(device)
        cloth = cloth.to(device)
        target = target.to(device)

        input_tensor = torch.cat([agnostic, cloth], dim=1)

        output = model(input_tensor)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (i+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/len(train_loader):.4f}, Time: {time.time()-start_time:.2f}s")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, "viton_unet_full_checkpoint.pth")

print("✅ Full model checkpoint saved.")

test_dataset = VITONTestDataset(root_dir='viton_dataset/ACGPN_TestData', transform=transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def visualize_output(model, dataset, idx=0):
    model.eval()
    agnostic, cloth, gt = dataset[idx]
    input_tensor = torch.cat([agnostic, cloth], dim=0).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()
        output_img = (output + 1) / 2  # Normalize to [0, 1]

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Agnostic")
    plt.imshow(TF.to_pil_image(agnostic))

    plt.subplot(1, 3, 2)
    plt.title("Cloth")
    plt.imshow(TF.to_pil_image(cloth))

    plt.subplot(1, 3, 3)
    plt.title("Generated Try-On")
    plt.imshow(TF.to_pil_image(output_img))
    plt.show()
