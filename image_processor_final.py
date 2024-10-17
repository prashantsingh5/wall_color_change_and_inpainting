import os
import numpy as np
import torch
from PIL import Image
import cv2
from segment_anything import build_sam, SamPredictor
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import load_image, predict
from GroundingDINO.groundingdino.models import build_model
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline
from torchvision.ops import box_convert

# Dictionary to map color names to their corresponding BGR values (OpenCV uses BGR)
COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "light red": (255, 102, 102),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "light blue": (173, 216, 230),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "lime": (0, 255, 0),
    "maroon": (128, 0, 0),
    "navy": (0, 0, 128),
    "olive": (128, 128, 0),
    "teal": (0, 128, 128),
    "silver": (192, 192, 192)
}

# Function to map color name to BGR value
def get_color_bgr(color_name):
    return COLOR_MAP.get(color_name.lower(), (0, 0, 255))  # Default to red if color not found

# Load the Grounding DINO model from HuggingFace Hub
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.to(device)
    model.eval()
    return model

# Function to show the mask on the image without blending
def show_mask(mask, image, color=(255, 0, 0)):  # Default color is red
    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8)
    mask_image[:, :] = color
    mask = mask.astype(np.uint8) * 255  # Convert mask to binary (0 or 255)
    result_image = np.where(mask[:, :, None] == 255, mask_image, image)
    return result_image

# Function to process the image, text prompt, and color for wall color change
def process_image(image_path, text_prompt, color_name="red", output_path="output_image.png", box_threshold=0.3, text_threshold=0.25):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Grounding DINO model
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device=device)

    # Load SAM model
    sam_checkpoint = r'C:\Users\pytorch\Desktop\try\weights\sam_vit_h_4b8939.pth'  # Replace with actual SAM checkpoint path
    sam_model = build_sam(checkpoint=sam_checkpoint)
    sam_model.to(device)
    sam_predictor = SamPredictor(sam_model)

    image_source, image = load_image(image_path)

    # Grounding DINO Prediction
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    sam_predictor.set_image(image_source)
    boxes = boxes.to(device)
    H, W, _ = image_source.shape
    scaling_tensor = torch.tensor([W, H, W, H], device=device)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * scaling_tensor
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    color_bgr = get_color_bgr(color_name)
    result_image = show_mask(masks[0][0].cpu().numpy(), image_source, color=color_bgr)

    result_image_pil = Image.fromarray(result_image)
    result_image_pil.save(output_path)
    print(f"Output image saved to {output_path}")
    return result_image_pil

# Function to load Stable Diffusion Inpainting pipeline
def load_inpainting_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    ).to("cuda")
    return pipe

# Function to generate masks for inpainting or keeping objects
def generate_masks_with_grounding(image_source, boxes, mask_type):
    image_source_array = np.array(image_source)
    h, w, _ = image_source_array.shape
    if len(boxes.shape) == 1:
        boxes = boxes[np.newaxis, :]
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source_array)
    if mask_type == "inpainting":
        for box in boxes_xyxy:
            x0, y0, x1, y1 = box
            mask[int(y0):int(y1), int(x0):int(x1), :] = 255  # White mask for inpainting
    return mask

# Function to detect objects and apply inpainting
def detect_objects_and_inpaint(image_path, box_threshold=0.45, text_threshold=0.25):
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device="cuda")

    pipe = load_inpainting_pipeline()

    default_objects = [
        "Bed frame", "headboard", "footboard", "dresser", "nightstand", "desk", "chair",
        "bedside table", "dressing table", "wardrobe", "closet organizer", "bookshelf",
        "ottoman or storage bench", "accent chair", "table lamps", "floor lamps",
        "ceiling light", "string lights", "sconce", "artwork", "pictures", "mirrors",
        "rugs", "curtains", "blinds", "plants", "candles and diffusers", "baskets",
        "closet", "drawers", "storage bins", "TV", "computer", "phone", "tablet",
        "headphones", "speakers", "bed", "wall", "window", "ceiling", "floor"
    ]

    caption = ", ".join(default_objects)
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = np.array(image_source)

    print("Detected objects:")
    for i, phrase in enumerate(phrases):
        print(f"Object {i + 1}: {phrase}")

    selected_indices = input("Enter the indices of the objects you want to inpaint (separated by spaces): ")
    selected_indices = [int(index) - 1 for index in selected_indices.split()]

    image_source_pil = Image.fromarray(image_source)

    for index in selected_indices:
        if index < len(phrases):
            mask = generate_masks_with_grounding(image_source_pil, boxes[index], "inpainting")
            image_resized = image_source_pil.resize((512, 512))
            mask_resized = Image.fromarray(mask).resize((512, 512))

            prompt = input(f"Enter the inpainting prompt for object {index + 1} ({phrases[index]}): ")
            inpainted_image = pipe(prompt=prompt, image=image_resized, mask_image=mask_resized).images[0]
            inpainted_image = inpainted_image.resize(image_source_pil.size)
            inpainted_image.show()
        else:
            print(f"Invalid index: {index}")

def detect_objects(image_path, box_threshold=0.30, text_threshold=0.20, return_objects=False):
    """
    Detects objects in the provided image and returns the list of detected objects.
    Parameters:
        image_path (str): The path to the image.
        box_threshold (float): Threshold for box detection.
        text_threshold (float): Threshold for text prompt matching.
        return_objects (bool): If True, return detected objects. Otherwise, print them.
    Returns:
        detected_objects (list): List of detected object phrases.
    """
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device="cuda")

    # Load and preprocess the image
    image_source = Image.open(image_path).convert('RGB')
    image_np = np.array(image_source)

    # Convert numpy array to PyTorch tensor and move to the correct device
    image = torch.from_numpy(image_np).permute(2, 0, 1).float().to("cuda")

    # List of default objects to detect
    default_objects = [
        "Bed frame", "headboard", "footboard", "dresser", "nightstand", "desk", "chair",
        "bedside table", "dressing table", "wardrobe", "closet organizer", "bookshelf",
        "ottoman or storage bench", "accent chair", "table lamps", "floor lamps",
        "ceiling light", "string lights", "sconce", "artwork", "pictures", "mirrors",
        "rugs", "curtains", "blinds", "plants", "candles and diffusers", "baskets",
        "closet", "drawers", "storage bins", "TV", "computer", "phone", "tablet",
        "headphones", "speakers", "bed", "wall", "window", "ceiling", "floor"
    ]

    caption = ", ".join(default_objects)

    # Perform object detection using the Grounding DINO model
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # Debugging: Print how many boxes and phrases are detected
    print(f"Number of boxes detected: {len(boxes)}")
    print(f"Number of phrases detected: {len(phrases)}")

    detected_objects = [phrase for phrase in phrases]

    if return_objects:
        return detected_objects

    # If not returning objects, print them
    print("Detected objects:")
    for i, phrase in enumerate(phrases):
        print(f"Object {i + 1}: {phrase}")

    return detected_objects

def process_image_for_inpainting(image_path, object_to_detect, inpaint_prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Grounding DINO model
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device=device)

    # Load SAM model
    sam_checkpoint = r'C:\Users\pytorch\Desktop\all projects\diffusion\SAM\weights\sam_vit_h_4b8939.pth'  # Replace with actual SAM checkpoint path
    sam_model = build_sam(checkpoint=sam_checkpoint)
    sam_model.to(device)
    sam_predictor = SamPredictor(sam_model)

    pipe = load_inpainting_pipeline()

    # Load and process image
    image_source, image = load_image(image_path)
    sam_predictor.set_image(image_source)

    # Detect object
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=object_to_detect,
        box_threshold=0.35,
        text_threshold=0.25
    )

    # Generate mask for inpainting
    mask = generate_masks_with_grounding(image_source, boxes[0], "inpainting")

    # Resize images for inpainting
    image_resized = Image.fromarray(image_source).resize((512, 512))
    mask_resized = Image.fromarray(mask).resize((512, 512))

    # Perform inpainting
    inpainted_image = pipe(prompt=inpaint_prompt, image=image_resized, mask_image=mask_resized).images[0]
    inpainted_image = inpainted_image.resize((image_source.shape[1], image_source.shape[0]))

    return inpainted_image