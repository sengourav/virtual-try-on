import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import StepLR
import random
import cv2

# Ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Improved dataset handling with proper data augmentation
class VITONDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, augment=False):
        """
        Enhanced dataset class with better error handling and data augmentation
        
        Args:
            root_dir: Root directory of the dataset
            mode: 'train' or 'test'
            transform: Transforms to apply to images
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.augment = augment
        
        # Check if directories exist
        img_dir = os.path.join(root_dir, f'{mode}_img')
        cloth_dir = os.path.join(root_dir, f'{mode}_color')
        label_dir = os.path.join(root_dir, f'{mode}_label')
        
        if not os.path.exists(img_dir) or not os.path.exists(cloth_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"One or more dataset directories not found in {root_dir}")
        
        # Get all image names
        self.image_names = []
        for f in sorted(os.listdir(img_dir)):
            if f.endswith('.jpg'):
                # Make sure corresponding files exist
                base_name = f.replace('_0.jpg', '')
                cloth_path = os.path.join(cloth_dir, f"{base_name}_1.jpg")
                label_path = os.path.join(label_dir, f"{base_name}_0.png")
                
                if os.path.exists(cloth_path) and os.path.exists(label_path):
                    self.image_names.append(base_name)
        
        print(f"Found {len(self.image_names)} valid samples in {mode} set")
    
    def __len__(self):
        return len(self.image_names)
    
    def _apply_augmentation(self, img, cloth, label):
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random brightness and contrast adjustment for person image
        if random.random() > 0.7:
            img = transforms.functional.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = transforms.functional.adjust_contrast(img, random.uniform(0.8, 1.2))
        
        # Random color jitter for clothing
        if random.random() > 0.7:
            cloth = transforms.functional.adjust_brightness(cloth, random.uniform(0.8, 1.2))
            cloth = transforms.functional.adjust_saturation(cloth, random.uniform(0.8, 1.2))
        
        return img, cloth, label
    
    def __getitem__(self, idx):
        base_name = self.image_names[idx]
        
        # Build file paths
        img_path = os.path.join(self.root_dir, f'{self.mode}_img', f"{base_name}_0.jpg")
        cloth_path = os.path.join(self.root_dir, f'{self.mode}_color', f"{base_name}_1.jpg")
        label_path = os.path.join(self.root_dir, f'{self.mode}_label', f"{base_name}_0.png")
        
        try:
            # Load images
            img = Image.open(img_path).convert('RGB').resize((192, 256))
            cloth = Image.open(cloth_path).convert('RGB').resize((192, 256))
            label = Image.open(label_path).convert('L').resize((192, 256), resample=Image.NEAREST)
            
            # Apply augmentation if enabled
            if self.augment and self.mode == 'train':
                img, cloth, label = self._apply_augmentation(img, cloth, label)
            
            # Convert label to numpy for processing
            img_np = np.array(img)
            label_np = np.array(label)
            
            # Create agnostic person image (remove upclothes → label 4)
            agnostic_np = img_np.copy()
            agnostic_np[label_np == 4] = [128, 128, 128]  # Grey out clothing region
            
            # Create cloth mask (binary mask of clothing)
            cloth_mask = (label_np == 4).astype(np.uint8) * 255
            cloth_mask_img = Image.fromarray(cloth_mask)
            
            # Apply transforms
            to_tensor = self.transform if self.transform else transforms.ToTensor()
            
            person_tensor = to_tensor(img)
            agnostic_tensor = to_tensor(Image.fromarray(agnostic_np))
            cloth_tensor = to_tensor(cloth)
            
            # Fix: Handle cloth mask properly
            if self.transform:
                # Convert to RGB for consistent channel handling
                cloth_mask_rgb = Image.fromarray(cloth_mask).convert('RGB')
                cloth_mask_tensor = to_tensor(cloth_mask_rgb)
            else:
                # Simple ToTensor() normalization for grayscale image
                cloth_mask_tensor = transforms.ToTensor()(cloth_mask_img)
                
                # If needed, expand to 3 channels
                if cloth_tensor.shape[0] == 3:  
                    cloth_mask_tensor = cloth_mask_tensor.expand(3, -1, -1)
            
            # One-hot encode the segmentation mask
            label_tensor = torch.from_numpy(label_np).long()
            
            sample = {
                'person': person_tensor,
                'agnostic': agnostic_tensor, 
                'cloth': cloth_tensor,
                'cloth_mask': cloth_mask_tensor,
                'label': label_tensor,
                'name': base_name
            }
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {base_name}: {e}")
            # Return a valid sample as fallback - get a different index
            return self.__getitem__((idx + 1) % len(self.image_names))
# class VITONDataset(Dataset):
#     def __init__(self, root_dir, mode='train', transform=None, augment=False):
#         """
#         Enhanced dataset class with better error handling and data augmentation
        
#         Args:
#             root_dir: Root directory of the dataset
#             mode: 'train' or 'test'
#             transform: Transforms to apply to images
#             augment: Whether to apply data augmentation
#         """
#         self.root_dir = root_dir
#         self.mode = mode
#         self.transform = transform
#         self.augment = augment
        
#         # Check if directories exist
#         img_dir = os.path.join(root_dir, f'{mode}_img')
#         cloth_dir = os.path.join(root_dir, f'{mode}_color')
#         label_dir = os.path.join(root_dir, f'{mode}_label')
        
#         if not os.path.exists(img_dir) or not os.path.exists(cloth_dir) or not os.path.exists(label_dir):
#             raise FileNotFoundError(f"One or more dataset directories not found in {root_dir}")
        
#         # Get all image names
#         self.image_names = []
#         for f in sorted(os.listdir(img_dir)):
#             if f.endswith('.jpg'):
#                 # Make sure corresponding files exist
#                 base_name = f.replace('_0.jpg', '')
#                 cloth_path = os.path.join(cloth_dir, f"{base_name}_1.jpg")
#                 label_path = os.path.join(label_dir, f"{base_name}_0.png")
                
#                 if os.path.exists(cloth_path) and os.path.exists(label_path):
#                     self.image_names.append(base_name)
        
#         print(f"Found {len(self.image_names)} valid samples in {mode} set")
    
#     def __len__(self):
#         return len(self.image_names)
    
#     def _apply_augmentation(self, img, cloth, label):
#         """Apply data augmentation"""
#         # Random horizontal flip
#         if random.random() > 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
#             label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
#         # Random brightness and contrast adjustment for person image
#         if random.random() > 0.7:
#             img = transforms.functional.adjust_brightness(img, random.uniform(0.8, 1.2))
#             img = transforms.functional.adjust_contrast(img, random.uniform(0.8, 1.2))
        
#         # Random color jitter for clothing
#         if random.random() > 0.7:
#             cloth = transforms.functional.adjust_brightness(cloth, random.uniform(0.8, 1.2))
#             cloth = transforms.functional.adjust_saturation(cloth, random.uniform(0.8, 1.2))
        
#         return img, cloth, label
    
#     def __getitem__(self, idx):
#         base_name = self.image_names[idx]
        
#         # Build file paths
#         img_path = os.path.join(self.root_dir, f'{self.mode}_img', f"{base_name}_0.jpg")
#         cloth_path = os.path.join(self.root_dir, f'{self.mode}_color', f"{base_name}_1.jpg")
#         label_path = os.path.join(self.root_dir, f'{self.mode}_label', f"{base_name}_0.png")
        
#         try:
#             # Load images
#             img = Image.open(img_path).convert('RGB').resize((192, 256))
#             cloth = Image.open(cloth_path).convert('RGB').resize((192, 256))
#             label = Image.open(label_path).convert('L').resize((192, 256), resample=Image.NEAREST)
            
#             # Apply augmentation if enabled
#             if self.augment and self.mode == 'train':
#                 img, cloth, label = self._apply_augmentation(img, cloth, label)
            
#             # Convert label to numpy for processing
#             img_np = np.array(img)
#             label_np = np.array(label)
            
#             # Create agnostic person image (remove upclothes → label 4)
#             agnostic_np = img_np.copy()
#             agnostic_np[label_np == 4] = [128, 128, 128]  # Grey out clothing region
            
#             # Create cloth mask (binary mask of clothing)
#             cloth_mask = (label_np == 4).astype(np.uint8) * 255
#             cloth_mask_img = Image.fromarray(cloth_mask)
            
#             # Apply transforms
#             to_tensor = self.transform if self.transform else transforms.ToTensor()
            
#             person_tensor = to_tensor(img)
#             agnostic_tensor = to_tensor(Image.fromarray(agnostic_np))
#             cloth_tensor = to_tensor(cloth)
            
#             # Fix: Ensure the cloth mask is properly processed to match expected dimensions
#             # First convert to Pillow Image with mode 'L' (grayscale)
#             cloth_mask_pil = Image.fromarray(cloth_mask, mode='L')
            
#             # Then apply the transform (which should normalize to [-1, 1] range)
#             if self.transform:
#                 # For custom transform that expects RGB input, convert grayscale to RGB
#                 cloth_mask_rgb = cloth_mask_pil.convert('RGB')
#                 cloth_mask_tensor = self.transform(cloth_mask_rgb)
#             else:
#                 # If using basic ToTensor, keep as grayscale but repeat to 3 channels if needed
#                 cloth_mask_tensor = transforms.ToTensor()(cloth_mask_pil)
                
#                 # If model expects 3 channels, repeat the single channel
#                 if cloth_tensor.shape[0] == 3:  # If cloth is RGB (3 channels)
#                     cloth_mask_tensor = cloth_mask_tensor.repeat(3, 1, 1)
            
#             # One-hot encode the segmentation mask
#             label_tensor = torch.from_numpy(label_np).long()
            
#             sample = {
#                 'person': person_tensor,
#                 'agnostic': agnostic_tensor, 
#                 'cloth': cloth_tensor,
#                 'cloth_mask': cloth_mask_tensor,
#                 'label': label_tensor,
#                 'name': base_name
#             }
            
#             return sample
            
#         except Exception as e:
#             print(f"Error loading sample {base_name}: {e}")
#             # Return a valid sample as fallback - get a different index
#             return self.__getitem__((idx + 1) % len(self.image_names))


# Improved U-Net with residual connections and attention
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Fixed: Change inplace ReLU to non-inplace
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # Fixed: Change inplace ReLU to non-inplace
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(PatchDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))  # Fixed: inplace=False
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
    
    def forward(self, img_A, img_B):
        # Concatenate image and condition
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
        
class ImprovedUNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(ImprovedUNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        
        # Bottleneck
        self.bottleneck = ResidualBlock(512)
        
        # Decoder
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),  # Fixed: inplace=False
            nn.Dropout(0.5)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),  # Fixed: inplace=False
            nn.Dropout(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)  # Fixed: inplace=False
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)  # Fixed: inplace=False
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        
        # Attention gates
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        b = self.bottleneck(e5)
        
        # Decoder with attention and skip connections
        d5 = self.dec5(b)
        d5 = torch.cat([self.att4(d5, e4), d5], dim=1)
        
        d4 = self.dec4(d5)
        d4 = torch.cat([self.att3(d4, e3), d4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([self.att2(d3, e2), d3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([self.att1(d2, e1), d2], dim=1)
        
        d1 = self.dec1(d2)
        
        return d1


# Discriminator network for adversarial training
class ImprovedUNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(ImprovedUNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False)  # Fixed: inplace=False
        )
        
        # Bottleneck
        self.bottleneck = ResidualBlock(512)
        
        # Decoder
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),  # Fixed: inplace=False
            nn.Dropout(0.5)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),  # Fixed: inplace=False
            nn.Dropout(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)  # Fixed: inplace=False
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)  # Fixed: inplace=False
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        
        # Attention gates
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        b = self.bottleneck(e5)
        
        # Decoder with attention and skip connections
        d5 = self.dec5(b)
        d5 = torch.cat([self.att4(d5, e4), d5], dim=1)
        
        d4 = self.dec4(d5)
        d4 = torch.cat([self.att3(d4, e3), d4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([self.att2(d3, e2), d3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([self.att1(d2, e1), d2], dim=1)
        
        d1 = self.dec1(d2)
        
        return d1


# Custom loss functions
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Import vgg here to avoid dependency at module level
        import torchvision.models as models
        
        # Load pretrained VGG but make sure to use non-inplace operations
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Replace inplace ReLU with non-inplace version
        for idx, module in enumerate(vgg):
            if isinstance(module, nn.ReLU):
                vgg[idx] = nn.ReLU(inplace=False)
        
        self.model = nn.Sequential()
        
        # Using feature layers
        feature_layers = [0, 2, 5, 10, 15, 20]
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
        for i in range(len(feature_layers)):
            self.model.add_module(f'layer_{i}', vgg[feature_layers[i]])
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        x_features = x
        y_features = y
        
        for i, layer in enumerate(self.model):
            x_features = layer(x_features)
            y_features = layer(y_features)
            
            if i in [0, 1, 2, 3, 4]:  # Only compute loss at specified layers
                loss += self.layer_weights[i] * self.criterion(x_features, y_features)
        
        return loss


# Training setup
def train_model(model_G, model_D=None, train_loader=None, val_loader=None, 
                num_epochs=50, device=None, use_gan=True):
    """
    Improved training function with GAN training, learning rate scheduler, and validation
    """
    torch.autograd.set_detect_anomaly(True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimizers
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    
    # Losses
    criterion_L1 = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss().to(device)
    
    # GAN setup
    if use_gan and model_D is not None:
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.5)
        criterion_GAN = nn.MSELoss()
    
    # Lists to store losses for plotting
    train_losses_G = []
    train_losses_D = [] if use_gan else None
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model_G.train()
        if use_gan and model_D is not None:
            model_D.train()
        
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0 if use_gan else None
        start_time = time.time()
        
        for i, sample in enumerate(train_loader):
            agnostic = sample['agnostic'].to(device)
            cloth = sample['cloth'].to(device)
            target = sample['person'].to(device)
            cloth_mask = sample['cloth_mask'].to(device)
            
            # Combine inputs
            input_tensor = torch.cat([agnostic, cloth], dim=1)
            
            # -----------------
            # Generator training
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake image
            fake_image = model_G(input_tensor)
            
            # Calculate L1 loss
            loss_L1 = criterion_L1(fake_image, target)
            
            # Calculate perceptual loss
            loss_perceptual = criterion_perceptual(fake_image, target)
            
            # Calculate total generator loss
            loss_G = loss_L1 + 0.1 * loss_perceptual
            
            # Add GAN loss if using adversarial training
            if use_gan and model_D is not None:
                # Adversarial loss (trick for stability: use 1s instead of 0.9)
                pred_fake = model_D(fake_image, cloth)
                target_real = torch.ones_like(pred_fake).to(device)
                loss_GAN = criterion_GAN(pred_fake, target_real)
                
                # Total generator loss with GAN component
                loss_G += 0.1 * loss_GAN
            
            # Backward pass and optimize generator
            loss_G.backward()
            optimizer_G.step()
            
            epoch_loss_G += loss_G.item()
            
            # -----------------
            # Discriminator training (if using GAN)
            # -----------------
            if use_gan and model_D is not None:
                optimizer_D.zero_grad()
                
                # Real loss
                pred_real = model_D(target, cloth)
                target_real = torch.ones_like(pred_real).to(device)
                loss_real = criterion_GAN(pred_real, target_real)
                
                # Fake loss
                pred_fake = model_D(fake_image.detach(), cloth)
                target_fake = torch.zeros_like(pred_fake).to(device)
                loss_fake = criterion_GAN(pred_fake, target_fake)
                
                # Total discriminator loss
                loss_D = (loss_real + loss_fake) / 2
                
                # Backward pass and optimize discriminator
                loss_D.backward()
                optimizer_D.step()
                
                epoch_loss_D += loss_D.item()
            
            # Print progress
            if (i+1) % 50 == 0:
                time_elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"G Loss: {loss_G.item():.4f}, "
                      f"{'D Loss: ' + f'{loss_D.item():.4f}, ' if use_gan else ''}"
                      f"Time: {time_elapsed:.2f}s")
        
        # Update learning rates
        scheduler_G.step()
        if use_gan and model_D is not None:
            scheduler_D.step()
        
        # Calculate average losses for this epoch
        avg_loss_G = epoch_loss_G / len(train_loader)
        train_losses_G.append(avg_loss_G)
        
        if use_gan:
            avg_loss_D = epoch_loss_D / len(train_loader)
            train_losses_D.append(avg_loss_D)
        
        # Validation
        if val_loader is not None:
            val_loss = validate_model(model_G, val_loader, device)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}, Train Loss G: {avg_loss_G:.4f}, "
                  f"{'Train Loss D: ' + f'{avg_loss_D:.4f}, ' if use_gan else ''}"
                  f"Val Loss: {val_loss:.4f}, "
                  f"Time: {time.time()-start_time:.2f}s")
        else:
            print(f"Epoch {epoch+1}, Train Loss G: {avg_loss_G:.4f}, "
                  f"{'Train Loss D: ' + f'{avg_loss_D:.4f}, ' if use_gan else ''}"
                  f"Time: {time.time()-start_time:.2f}s")
        
        # Save model checkpoint periodically
        if (epoch+1) % 5 == 0:
            save_checkpoint(model_G, model_D, optimizer_G, optimizer_D if use_gan else None, 
                           epoch, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Visualize some results
        if (epoch+1) % 5 == 0:
            visualize_results(model_G, val_loader, device, epoch+1)
    
    # Plot training losses
    plot_losses(train_losses_G, train_losses_D, val_losses)
    
    return model_G, model_D


def validate_model(model, val_loader, device):
    """Validate the model on validation set"""
    model.eval()
    val_loss = 0.0
    criterion = nn.L1Loss()
    
    with torch.no_grad():
        for sample in val_loader:
            agnostic = sample['agnostic'].to(device)
            cloth = sample['cloth'].to(device)
            target = sample['person'].to(device)
            
            input_tensor = torch.cat([agnostic, cloth], dim=1)
            output = model(input_tensor)
            
            loss = criterion(output, target)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def visualize_results(model, dataloader, device, epoch):
    """Visualize generated try-on results"""
    model.eval()
    
    # Get a batch of samples
    for i, sample in enumerate(dataloader):
        if i >= 1:  # Only visualize first batch
            break
            
        with torch.no_grad():
            agnostic = sample['agnostic'].to(device)
            cloth = sample['cloth'].to(device)
            target = sample['person'].to(device)
            
            input_tensor = torch.cat([agnostic, cloth], dim=1)
            output = model(input_tensor)
            
            # Convert tensors for visualization
            imgs = []
            for j in range(min(4, output.size(0))):  # Show max 4 examples
                person_img = (target[j].cpu().permute(1, 2, 0).numpy() + 1) / 2
                agnostic_img = (agnostic[j].cpu().permute(1, 2, 0).numpy() + 1) / 2
                cloth_img = (cloth[j].cpu().permute(1, 2, 0).numpy() + 1) / 2
                output_img = (output[j].cpu().permute(1, 2, 0).numpy() + 1) / 2
                
                # Combine images for visualization
                row1 = np.hstack([agnostic_img, cloth_img])
                row2 = np.hstack([output_img, person_img])
                combined = np.vstack([row1, row2])
                
                imgs.append(combined)
            
            # Create figure
            fig, axs = plt.subplots(1, len(imgs), figsize=(15, 5))
            if len(imgs) == 1:
                axs = [axs]
                
            for j, img in enumerate(imgs):
                axs[j].imshow(img)
                axs[j].set_title(f"Sample {j+1}")
                axs[j].axis('off')
            
            fig.suptitle(f"Epoch {epoch} Results", fontsize=16)
            plt.tight_layout()
            
            # Save figure
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/epoch_{epoch}_samples.png')
            plt.close()


def plot_losses(train_losses_G, train_losses_D=None, val_losses=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_G, label='Generator Loss')
    
    if train_losses_D:
        plt.plot(train_losses_D, label='Discriminator Loss')
    
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/loss_plot.png')
    plt.close()


def save_checkpoint(model_G, model_D=None, optimizer_G=None, optimizer_D=None, epoch=None, filename="checkpoint.pth"):
    """Save model checkpoint"""
    os.makedirs('checkpoints', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_G_state_dict': model_G.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G else None,
    }
    
    if model_D is not None:
        checkpoint['model_D_state_dict'] = model_D.state_dict()
    
    if optimizer_D is not None:
        checkpoint['optimizer_D_state_dict'] = optimizer_D.state_dict()
    
    torch.save(checkpoint, f'checkpoints/{filename}')


def load_checkpoint(model_G, model_D=None, optimizer_G=None, optimizer_D=None, filename="checkpoint.pth"):
    """Load model checkpoint"""
    checkpoint = torch.load(f'checkpoints/{filename}')
    
    model_G.load_state_dict(checkpoint['model_G_state_dict'])
    
    if optimizer_G and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    
    if model_D is not None and 'model_D_state_dict' in checkpoint:
        model_D.load_state_dict(checkpoint['model_D_state_dict'])
    
    if optimizer_D is not None and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    return checkpoint.get('epoch', 0)


# Test function
def test_model(model, test_loader, device, result_dir='test_results'):
    """Generate and save test results"""
    model.eval()
    os.makedirs(result_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            agnostic = sample['agnostic'].to(device)
            cloth = sample['cloth'].to(device)
            target = sample['person'].to(device)
            name = sample['name'][0]  # Get sample name
            
            # Generate try-on result
            input_tensor = torch.cat([agnostic, cloth], dim=1)
            output = model(input_tensor)
            
            # Save images
            output_img = (output[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            target_img = (target[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            agnostic_img = (agnostic[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            cloth_img = (cloth[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            
            # Save individual images
            plt.imsave(f'{result_dir}/{name}_output.png', output_img)
            plt.imsave(f'{result_dir}/{name}_target.png', target_img)
            
            # Save comparison grid
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))
            ax[0, 0].imshow(agnostic_img)
            ax[0, 0].set_title('Person (w/o clothes)')
            ax[0, 0].axis('off')
            
            ax[0, 1].imshow(cloth_img)
            ax[0, 1].set_title('Clothing Item')
            ax[0, 1].axis('off')
            
            ax[1, 0].imshow(output_img)
            ax[1, 0].set_title('Generated Result')
            ax[1, 0].axis('off')
            
            ax[1, 1].imshow(target_img)
            ax[1, 1].set_title('Ground Truth')
            ax[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{result_dir}/{name}_comparison.png')
            plt.close()
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_loader)} test samples")