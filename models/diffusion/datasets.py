import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class VITONBaseDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(os.path.join(root_dir, f'{split}_img')) if f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.image_names)

    def _load_images(self, base_name):
        img = Image.open(os.path.join(self.root_dir, f'{self.split}_img', f"{base_name}_0.jpg")).resize((192, 256))
        cloth = Image.open(os.path.join(self.root_dir, f'{self.split}_color', f"{base_name}_1.jpg")).resize((192, 256))
        label = Image.open(os.path.join(self.root_dir, f'{self.split}_label', f"{base_name}_0.png")).resize((192, 256), resample=Image.NEAREST)
        return img, cloth, label

    def __getitem__(self, idx):
        base_name = self.image_names[idx].replace('_0.jpg', '')
        img, cloth, label = self._load_images(base_name)

        img_np = np.array(img)
        label_np = np.array(label)
        agnostic_np = img_np.copy()
        agnostic_np[label_np == 4] = [128, 128, 128]

        to_tensor = self.transform if self.transform else transforms.ToTensor()
        return to_tensor(Image.fromarray(agnostic_np)), to_tensor(cloth), to_tensor(img)

class VITONTrainDataset(VITONBaseDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, split="train", transform=transform)

class VITONTestDataset(VITONBaseDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, split="test", transform=transform)
