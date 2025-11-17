import torch
import timm
import numpy as np
import cv2
import argparse
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image

CLASS_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6
}

def load_image(data_dir: Path, image_id: str):
    # Try to find image in either part_1 or part_2
    for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = data_dir / sub / f"{image_id}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find {image_id}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../archive", help="Path to HAM10000 data")
    ap.add_argument("--model_path", default="../outputs/efficientnet_b0_img224_seed42/checkpoints/best.pt", 
                    help="Path to trained model checkpoint")
    ap.add_argument("--backbone", default="efficientnet_b0", help="Model backbone")
    ap.add_argument("--image_ids", nargs="+", default=["ISIC_0027419", "ISIC_0025030"], 
                    help="Image IDs to visualize")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.backbone, pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get target layer for GradCAM (last conv layer)
    if "efficientnet" in args.backbone:
        target_layers = [model.conv_head]
    elif "densenet" in args.backbone:
        target_layers = [model.features[-1]]
    else:
        target_layers = [model.stages[-1]]

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process each image
    for image_id in args.image_ids:
        try:
            img_path = load_image(data_dir, image_id)
            
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            
            # For GradCAM overlay, we need the original RGB image normalized to [0,1]
            rgb_img = cv2.imread(str(img_path))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255.0
            
            # Prepare input tensor
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()
            
            # Generate GradCAM
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            # Overlay CAM on image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Save result
            output_path = out_dir / f"gradcam_{image_id}.png"
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), cam_image_bgr)
            
            print(f"Saved GradCAM for {image_id} (predicted class: {pred_class}) -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")

if __name__ == "__main__":
    main()
