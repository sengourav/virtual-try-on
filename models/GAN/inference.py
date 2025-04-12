import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from improved_viton import ImprovedUNetGenerator, set_seed
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

def preprocess_image(img_path, size=(192, 256)):
    img = Image.open(img_path).convert('RGB').resize(size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)

def segment_person(img_path, size=(192, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "matei-dorian/segformer-b5-finetuned-human-parsing"
    ).to(device).eval()

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    seg_map_resized = cv2.resize(seg_map.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

    clothing_labels = [4, 5, 6, 7, 8, 16]
    segmentation = np.zeros_like(seg_map_resized)
    for label in clothing_labels:
        segmentation[seg_map_resized == label] = 1

    return segmentation

def create_agnostic_image(img_path, segmentation, size=(192, 256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    agnostic = img.copy()
    agnostic[segmentation == 1] = [128, 128, 128]
    return agnostic

def inference(model_path, person_img_path, clothing_img_path, output_path='try_on_result.png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImprovedUNetGenerator(in_channels=6, out_channels=3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_G_state_dict'])
    model.eval()
    print("Model loaded successfully")

    print("Segmenting person image...")
    segmentation = segment_person(person_img_path)
    agnostic = create_agnostic_image(person_img_path, segmentation)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    agnostic_tensor = transform(Image.fromarray(agnostic)).unsqueeze(0).to(device)

    print("Processing clothing image...")
    cloth = cv2.imread(clothing_img_path)
    cloth = cv2.resize(cloth, (192, 256))
    cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)
    cloth_tensor = transform(Image.fromarray(cloth)).unsqueeze(0).to(device)

    print("Generating virtual try-on result...")
    with torch.no_grad():
        input_tensor = torch.cat([agnostic_tensor, cloth_tensor], dim=1)
        output = model(input_tensor)

    output_img = output[0].cpu().permute(1, 2, 0).numpy()
    output_img = (output_img + 1) / 2
    output_img = np.clip(output_img, 0, 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    person_img = cv2.imread(person_img_path)
    person_img = cv2.resize(person_img, (192, 256))
    person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    plt.imshow(person_img)
    plt.title("Original Person")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cloth)
    plt.title("Clothing Item")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(agnostic)
    plt.title("Person without Clothing")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(output_img)
    plt.title("Try-On Result")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")

    result_img = (output_img * 255).astype(np.uint8)
    Image.fromarray(result_img).save('result_only.png')

    return output_img

def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--person', type=str, required=True, help='Path to the person image')
    parser.add_argument('--clothing', type=str, required=True, help='Path to the clothing image')
    parser.add_argument('--output', type=str, default='try_on_result.png', help='Path to save the output image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)
    inference(args.model, args.person, args.clothing, args.output)

if __name__ == "__main__":
    main()
