# main.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import VITONTrainDataset, VITONTestDataset
from model import UNet, DiffusionModel

def generate_samples(diffusion_model, test_loader, device, num_samples=5, epoch=0):
    diffusion_model.eval()
    os.makedirs("samples", exist_ok=True)

    with torch.no_grad():
        for i, (agnostic, cloth, target) in enumerate(test_loader):
            if i >= num_samples:
                break

            agnostic, cloth, target = agnostic.to(device), cloth.to(device), target.to(device)
            condition = torch.cat([agnostic, cloth], dim=1)
            sample_shape = (1, 3, 256, 192)
            generated = diffusion_model.sample(condition, sample_shape, device)
            generated = (generated + 1) / 2

            to_pil = transforms.ToPILImage()
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(to_pil(agnostic[0].cpu()))
            plt.title("Person")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(to_pil(cloth[0].cpu()))
            plt.title("Clothing")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(to_pil(generated[0].cpu()))
            plt.title("Generated Try-On")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"samples/sample_epoch{epoch}_idx{i}.png")
            plt.close()
def visualize_sample(dataset, index=0):
    agnostic, cloth, target = dataset[index]
    to_pil = transforms.ToPILImage()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(to_pil(agnostic))
    plt.title("Agnostic Person Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(to_pil(cloth))
    plt.title("Clothing Item")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(to_pil(target))
    plt.title("Target Try-On Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
  
def main():
    batch_size = 4
    num_workers = 4
    num_epochs = 50
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = VITONTrainDataset(root_dir='viton_dataset/ACGPN_TrainData', transform=transform)
    test_dataset = VITONTestDataset(root_dir='viton_dataset/ACGPN_TestData', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    unet = UNet(in_channels=6, out_channels=3, model_channels=64)
    diffusion = DiffusionModel(unet, num_timesteps=1000).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        diffusion.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (agnostic, cloth, target) in progress_bar:
            agnostic, cloth, target = agnostic.to(device), cloth.to(device), target.to(device)
            target = 2 * target - 1
            condition = torch.cat([agnostic, cloth], dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (agnostic.size(0),), device=device).long()

            optimizer.zero_grad()
            loss = diffusion.p_losses(target, condition, t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss / len(train_loader):.6f}")

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader),
            }, f"diffusion_viton_checkpoint_epoch_{epoch+1}.pth")

            if epoch % 10 == 0:
                generate_samples(diffusion, test_loader, device, num_samples=5, epoch=epoch)

if __name__ == "__main__":
    main()
