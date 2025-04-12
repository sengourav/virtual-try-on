import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import zipfile
import numpy as np
from tqdm import tqdm

# Import our improved modules
from improved_viton import (
    VITONDataset, 
    ImprovedUNetGenerator, 
    PatchDiscriminator,
    train_model, 
    test_model, 
    set_seed,
    visualize_results,
    save_checkpoint,
    load_checkpoint
)

def extract_dataset(dataset_path, extract_dir):
    """Extract the dataset if needed"""
    if not os.path.exists(extract_dir):
        print(f"Extracting dataset to {extract_dir}...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print("Dataset Structure:")
        for folder in os.listdir(extract_dir):
            print("-", folder)
    else:
        print(f"Dataset directory {extract_dir} already exists.")

def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On Training Script')
    parser.add_argument('--dataset_path', type=str, default="/content/drive/MyDrive/virtual-try-on/dataset.zip", 
                        help='Path to the dataset zip file')
    parser.add_argument('--extract_dir', type=str, default="viton_dataset", 
                        help='Directory to extract the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--use_gan', action='store_true', help='Use GAN training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true', help='Run only testing, no training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Extract dataset if needed
    extract_dataset(args.dataset_path, args.extract_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Create datasets
    train_dataset = VITONDataset(
        root_dir=os.path.join(args.extract_dir, 'ACGPN_TrainData'),
        mode='train',
        transform=transform,
        augment=True
    )
    
    test_dataset = VITONDataset(
        root_dir=os.path.join(args.extract_dir, 'ACGPN_TestData'),
        mode='test',
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Create validation loader (using a subset of test data)
    val_dataset = torch.utils.data.Subset(test_dataset, range(0, min(100, len(test_dataset))))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create models
    model_G = ImprovedUNetGenerator(in_channels=6, out_channels=3).to(device)
    
    if args.use_gan:
        model_D = PatchDiscriminator(in_channels=6).to(device)
    else:
        model_D = None
    
    # Create optimizers
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr, betas=(0.5, 0.999)) if model_D else None
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume and args.checkpoint:
        start_epoch = load_checkpoint(model_G, model_D, optimizer_G, optimizer_D, args.checkpoint)
        print(f"Resumed from checkpoint, starting at epoch {start_epoch + 1}")
    
    # Testing only
    if args.test_only:
        if args.checkpoint:
            load_checkpoint(model_G, None, None, None, args.checkpoint)
            print(f"Loaded checkpoint for testing")
        else:
            print("Warning: Testing without loading a checkpoint!")
        
        print("Running test...")
        test_model(model_G, test_loader, device, result_dir='test_results')
        print("Testing done!")
        return
    
    # Training
    print("Starting training...")
    model_G, model_D = train_model(
        model_G=model_G,
        model_D=model_D,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        use_gan=args.use_gan
    )
    
    # Save final model
    save_checkpoint(model_G, model_D, optimizer_G, optimizer_D, args.epochs, "final_model.pth")
    
    # Run final test
    print("Running final test...")
    test_model(model_G, test_loader, device, result_dir='final_results')
    
    print("Training and testing completed!")

if __name__ == "__main__":
    main()
