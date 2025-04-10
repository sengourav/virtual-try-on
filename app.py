# import streamlit as st
# from PIL import Image
# import torch
# import numpy as np
# from torchvision import transforms
# import torch.nn as nn
# from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
# # import cv2
# import os
# import gdown
# import zipfile
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #download segformer
# model_dir = "segformer"
# zip_path = "segformer.zip"
# url = "https://drive.google.com/uc?export=download&id=1hpwO8RximyjmILpvsjOLJRi2Wq1AxM67"

# if not os.path.exists(model_dir):
#     os.system(f"wget --no-check-certificate '{url}' -O {zip_path}")
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(".")

# # Initialize processor and model
# processor = AutoImageProcessor.from_pretrained(model_dir,local_files_only=True)
# parser_model = SegformerForSemanticSegmentation.from_pretrained(model_dir,local_files_only=True).to("cuda").eval()


# #download try-on model
# MODEL_PATH = "viton_unet_full_checkpoint.pth"
# GDRIVE_ID = "1SNdyZM2IWQ6L85VBH-B4JhkGKwEcSq2T"
# GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

# # Download only if not already present
# if not os.path.exists(MODEL_PATH):
#     with st.spinner("Downloading model weights..."):
#         gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# # Load segmentation model
# # processor = AutoImageProcessor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
# # parser_model = SegformerForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing").to(device).eval()
# # Load your trained try-on model


# class UNetGenerator(nn.Module):
#     def __init__(self, in_channels=6, out_channels=3):
#         super(UNetGenerator, self).__init__()

#         def block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, 4, 2, 1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )

#         def up_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )

#         self.down1 = block(in_channels, 64)
#         self.down2 = block(64, 128)
#         self.down3 = block(128, 256)
#         self.down4 = block(256, 512)

#         self.up1 = up_block(512, 256)
#         self.up2 = up_block(512, 128)
#         self.up3 = up_block(256, 64)
#         self.up4 = nn.Sequential(
#             nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)

#         u1 = self.up1(d4)
#         u2 = self.up2(torch.cat([u1, d3], dim=1))
#         u3 = self.up3(torch.cat([u2, d2], dim=1))
#         u4 = self.up4(torch.cat([u3, d1], dim=1))
#         return u4


# # Initialize model
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tryon_model = UNetGenerator().to(device)

# # Load checkpoint
# checkpoint = torch.load("viton_unet_full_checkpoint.pth", map_location=device)
# tryon_model.load_state_dict(checkpoint['model_state_dict'])
# tryon_model.eval()

# # Label colors for visualization (for simplicity, a random colormap)
# LABEL_COLORS = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)

# # Resize and normalize transforms
# img_transform = transforms.Compose([
#     transforms.Resize((256, 192)),
#     transforms.ToTensor()
# ])

# @st.cache_data
# def get_segmentation(image: Image.Image):
#     inputs = processor(images=image, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         outputs = parser_model(**inputs)
#     logits = outputs.logits
#     predicted = torch.argmax(logits, dim=1)[0].cpu().numpy()
#     return predicted

# def generate_agnostic(image: Image.Image, segmentation):
#     img_np = np.array(image.resize((192, 256)))
#     agnostic_np = img_np.copy()
#     segmentation_resized = cv2.resize(segmentation.astype(np.uint8), (192, 256), interpolation=cv2.INTER_NEAREST)
#     agnostic_np[segmentation_resized == 4] = [128, 128, 128]  # Remove upper clothes
#     return Image.fromarray(agnostic_np)

# def generate_tryon_output(agnostic_img, cloth_img):
#     # Preprocess inputs
#     agnostic_tensor = img_transform(agnostic_img).unsqueeze(0).to("cuda")
#     cloth_tensor = img_transform(cloth_img).unsqueeze(0).to("cuda")
#     input_tensor = torch.cat([agnostic_tensor, cloth_tensor], dim=1)
#     # Run inference
#     with torch.no_grad():
#         output = tryon_model(input_tensor)
#     output_img = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
#     output_img = (output_img * 255).astype(np.uint8)
#     return Image.fromarray(output_img)

# # Streamlit UI
# st.title("👕 Virtual Try-On App (2D Non-Warping UNet)")

# person_file = st.file_uploader("Upload a person image", type=["jpg", "png", "jpeg"])
# cloth_file = st.file_uploader("Upload a cloth image", type=["jpg", "png", "jpeg"])

# if person_file and cloth_file:
#     person_img = Image.open(person_file).convert("RGB")
#     cloth_img = Image.open(cloth_file).convert("RGB")

#     st.image([person_img, cloth_img], caption=["Person", "Cloth"], width=200)

#     seg_map = get_segmentation(person_img)
#     agnostic_img = generate_agnostic(person_img, seg_map)

#     st.subheader("Agnostic Image (Torso-Masked)")
#     w, h = agnostic_img.size
#     st.image(agnostic_img)


#     output_img = generate_tryon_output(agnostic_img, cloth_img)
    
#     st.subheader("Virtual Try-On Result")
#     w, h = output_img.size
#     st.image(output_img, caption="Virtual Try-On Result")


import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Google Drive Folder IDs -----------------
UNET_MODEL_ID = "1SNdyZM2IWQ6L85VBH-B4JhkGKwEcSq2T"
PARSER_MODEL_FOLDER_ID = "1uOAJpgvJ815xFs3Ihn8RhRjNs9GTxh4f"
PARSER_PROCESSOR_FOLDER_ID = "1xsY9oNwoboJQozVPY8OgaChSmv93aL98"

# ----------------- Local Paths -----------------
UNET_PATH = "viton_unet_full_checkpoint.pth"
PARSER_MODEL_DIR = "human_parser_model"
PARSER_PROCESSOR_DIR = "human_parser_processor"

# ----------------- Google Drive Folder Downloader -----------------
def download_folder(folder_id: str, output_path: str):
    try:
        import gdown
        gdown.download_folder(id=folder_id, output=output_path, quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"❌ Failed to download folder: {output_path}\n{e}")

def download_file(file_id: str, output_path: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    subprocess.run(["wget", "-O", output_path, url], check=True)

# ----------------- Download Required Files -----------------
if not os.path.exists(UNET_PATH):
    with st.spinner("Downloading UNet model..."):
        download_file(UNET_MODEL_ID, UNET_PATH)

if not os.path.exists(PARSER_MODEL_DIR):
    with st.spinner("Downloading Human Parser Model folder..."):
        download_folder(PARSER_MODEL_FOLDER_ID, PARSER_MODEL_DIR)

if not os.path.exists(PARSER_PROCESSOR_DIR):
    with st.spinner("Downloading Human Parser Processor folder..."):
        download_folder(PARSER_PROCESSOR_FOLDER_ID, PARSER_PROCESSOR_DIR)

# ----------------- Load Human Parser Model -----------------
processor = AutoImageProcessor.from_pretrained(PARSER_PROCESSOR_DIR)
parser_model = SegformerForSemanticSegmentation.from_pretrained(PARSER_MODEL_DIR).to(device).eval()

# ----------------- UNet Definition -----------------
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

# ----------------- Load UNet Model -----------------
tryon_model = UNetGenerator().to(device)
checkpoint = torch.load(UNET_PATH, map_location=device)
tryon_model.load_state_dict(checkpoint['model_state_dict'])
tryon_model.eval()

# ----------------- Image Preprocessing -----------------
img_transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor()
])

@st.cache_data
def get_segmentation(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = parser_model(**inputs)
    logits = outputs.logits
    predicted = torch.argmax(logits, dim=1)[0].cpu().numpy()
    return predicted

def generate_agnostic(image: Image.Image, segmentation):
    img_np = np.array(image.resize((192, 256)))
    agnostic_np = img_np.copy()
    segmentation_resized = cv2.resize(segmentation.astype(np.uint8), (192, 256), interpolation=cv2.INTER_NEAREST)
    agnostic_np[segmentation_resized == 4] = [128, 128, 128]  # Upper-clothes class
    return Image.fromarray(agnostic_np)

def generate_tryon_output(agnostic_img, cloth_img):
    agnostic_tensor = img_transform(agnostic_img).unsqueeze(0).to(device)
    cloth_tensor = img_transform(cloth_img).unsqueeze(0).to(device)
    input_tensor = torch.cat([agnostic_tensor, cloth_tensor], dim=1)

    with torch.no_grad():
        output = tryon_model(input_tensor)
    output_img = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_img = (output_img * 255).astype(np.uint8)
    return Image.fromarray(output_img)

# ----------------- Streamlit UI -----------------
st.title("👕 Virtual Try-On (UNet + Human Parser)")

person_file = st.file_uploader("Upload a person image", type=["jpg", "jpeg", "png"])
cloth_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])

if person_file and cloth_file:
    person_img = Image.open(person_file).convert("RGB")
    cloth_img = Image.open(cloth_file).convert("RGB")

    st.image([person_img, cloth_img], caption=["Person", "Cloth"], width=200)

    seg_map = get_segmentation(person_img)
    agnostic_img = generate_agnostic(person_img, seg_map)

    st.subheader("Agnostic Image (Torso Masked)")
    st.image(agnostic_img)

    output_img = generate_tryon_output(agnostic_img, cloth_img)

    st.subheader("Virtual Try-On Output")
    st.image(output_img, caption="Try-On Result")


