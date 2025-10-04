# ==============================================================================
# 1. INSTALL DEPENDENCIES (run once from terminal, not inside script)
# ==============================================================================
# In PowerShell or CMD, run:
#   pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
#   pip install "git+https://github.com/IDEA-Research/GroundingDINO.git"
#   pip install supervision opencv-python pillow matplotlib requests torch torchvision

# ==============================================================================
# 2. IMPORTS AND SETUP
# ==============================================================================
import os
import torch
import cv2
import supervision as sv
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import subprocess

# Grounding DINO imports
from groundingdino.util.inference import Model as GroundingDINOModel

# SAM 2 imports
from segment_anything import sam_model_registry, SamPredictor

print("Imports complete.")

# ==============================================================================
# 3. LOAD MODELS
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- GroundingDINO Setup ---
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"

if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
    print("Downloading GroundingDINO checkpoint...")
    subprocess.run([
        "curl", "-L", "-o", GROUNDING_DINO_CHECKPOINT_PATH,
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ])

print("Loading GroundingDINO model...")
grounding_dino_model = GroundingDINOModel(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    device=str(DEVICE)
)
print("GroundingDINO model loaded.")

# --- SAM 2 Setup ---
SAM2_CHECKPOINT_PATH = "sam2_hiera_large.pt"
MODEL_TYPE = "hiera_l"

if not os.path.exists(SAM2_CHECKPOINT_PATH):
    print("Downloading SAM 2 checkpoint...")
    subprocess.run([
        "curl", "-L", "-o", SAM2_CHECKPOINT_PATH,
        "https://dl.fbaipublicfiles.com/segment_anything_2/032424/sam2_hiera_large.pt"
    ])

print("Loading SAM 2 model...")
sam2 = sam_model_registry[MODEL_TYPE](checkpoint=SAM2_CHECKPOINT_PATH)
sam2.to(device=DEVICE)
sam_predictor = SamPredictor(sam2)
print("SAM 2 model loaded.")

# ==============================================================================
# 4. DEFINE THE PIPELINE
# ==============================================================================
def segment_image_with_text(image_url: str, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_pil = Image.open(BytesIO(response.content)).convert("RGB")
        image_np = np.array(image_pil)
    except requests.exceptions.RequestException as e:
        print(f"Error loading image: {e}")
        return

    print(f"Running GroundingDINO with prompt: '{text_prompt}'")
    detections = grounding_dino_model.predict_with_classes(
        image=image_np,
        classes=[text_prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    detections = detections[detections.class_id != None]
    if len(detections.xyxy) == 0:
        print("No objects detected for this prompt.")
        return

    print(f"Detected {len(detections.xyxy)} instance(s) of '{text_prompt}'.")

    print("Running SAM 2 to generate masks...")
    sam_predictor.set_image(image_np)
    input_boxes = torch.tensor(detections.xyxy, device=DEVICE)

    masks, scores, logits = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=input_boxes,
        multimask_output=False,
    )

    masks_np = masks.squeeze(1).cpu().numpy()

    print("Visualizing results...")
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated_boxes = box_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated_masks = mask_annotator.annotate(scene=image_np.copy(), detections=sv.Detections(xyxy=detections.xyxy, mask=masks_np))

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(image_pil); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(annotated_boxes); axes[1].set_title("GroundingDINO Detections"); axes[1].axis('off')
    axes[2].imshow(annotated_masks); axes[2].set_title("SAM 2 Segmentation Mask"); axes[2].axis('off')
    plt.tight_layout(); plt.show()

# ==============================================================================
# 5. RUN THE DEMO
# ==============================================================================
IMAGE_URL = "https://media.roboflow.com/notebooks/examples/dog.jpeg"
TEXT_PROMPT = "the dog's face"

segment_image_with_text(IMAGE_URL, TEXT_PROMPT)

IMAGE_URL_2 = "https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?q=80&w=2069"
TEXT_PROMPT_2 = "a laptop"

segment_image_with_text(IMAGE_URL_2, TEXT_PROMPT_2)
