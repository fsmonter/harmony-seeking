import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
from fastapi.staticfiles import StaticFiles
from segment_anything import SamPredictor, sam_model_registry
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import io

app = FastAPI()
app.mount("/images", StaticFiles(directory="./images"), name="images")

output_dir = Path("./images")
output_dir.mkdir(exist_ok=True)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

patterns = [
    {"name": "Window Place", "description": "Create comfortable sitting near natural light."},
    {"name": "Thick Walls", "description": "Use thick walls to create boundaries and distinct rooms."},
    {"name": "Positive Outdoor Space", "description": "Outdoor spaces should have positive shape."},
    {"name": "Boundaries", "description": "Use boundaries to clearly separate areas."},
    {"name": "Local Symmetry", "description": "Symmetrical local centers enhance harmony."},
    {"name": "Indoor Sunlight", "description": "If the right rooms are facing south, a house is bright and sunny and cheerful; if the wrong rooms are facing south, the house is dark and gloomy."},
    {"name": "Site Repair", "description": "On no account place buildings in the places which are most beautiful. In fact, do the opposite. Consider the site and its buildings as a single living eco-system. Leave those areas that are the most precious, beautiful, comfortable, and healthy as they are, and build new structures in those parts of the site which are the least pleasant now."},
    {"name": "Workspace Enclosure", "description": "People cannot work effectively if their workspace is too enclosed or too exposed. A good workspace strikes the balance."},
    {"name": "The Fire", "description": "Build the fire in a common space. Perhaps in the kitchen. Where it provides a natural focus for talk and dreams and thought. Adjust the location until it knits together the social spaces and rooms around it, giving them each a glimpse of the fire; and make a window or some other focus to sustain the place during the times when the fire is out."},
    {"name": "House Cluster", "description": "People will not feel comfortable in their houses unless a group of houses forms a cluster, with the public land between them jointly owned by all the householders. Therefore: Arrange houses to form very rough, but identifiable clusters of 8 to 12 households around some common land and paths. Arrange the clusters so that anyone can walk through them, without feeling like a trespasser."},
    {"name": "Natural Doors and Windows", "description": "Finding the right position for a window or a door is a subtle matter. But there are very few ways of building which take this into consideration. Therefore: On no account use standard doors or windows. Make each window a different size, according to its place. Do not fix the exact position or size of the door and window frames until the rough framing of the room has actually been built, and you can really stand inside the room and judge, by eye, exactly where you want to put them, and how big you want them. When you decide, mark the openings with strings. Make the windows smaller and smaller, as you go higher in the building."},
    {"name": "Couple's Realm", "description": "The presence of children in a family often destroys the closeness and the special privacy which a man and wife need together. Therefore: Make a special part of the house distinct from the common areas and all the children's rooms, where the man and woman of the house can be together in private. Give this place a quick path to the children's rooms, but, at all costs, make it a distinctly separate realm"},
    # Add more patterns...
]

pattern_elements = {
    "Window Place": [
        "windows", "natural light", "seating area", "comfortable chairs",
        "view outside", "indoor plants", "reading nook"
    ],
    "Thick Walls": [
        "thick walls", "solid boundaries", "room separation", "structural elements",
        "wall depth", "distinct spaces", "insulation", "masonry"
    ],
    "Positive Outdoor Space": [
        "outdoor areas", "garden space", "patio", "courtyard",
        "landscaping", "defined boundaries", "natural elements", "outdoor seating"
    ],
    "Boundaries": [
        "walls", "fences", "hedges", "space division",
        "distinct areas", "transitions", "thresholds", "borders"
    ],
    "Local Symmetry": [
        "balanced elements", "symmetrical features", "aligned windows",
        "centered doorways", "matching details", "proportional spaces"
    ],
    "Indoor Sunlight": [
        "windows", "sunlight", "bright space", "southern exposure",
        "natural lighting", "warm glow", "shadow patterns"
    ],
    "Site Repair": [
        "natural landscape", "existing features", "preserved areas",
        "integrated design", "environmental harmony", "ecological balance"
    ],
    "Workspace Enclosure": [
        "walls", "partial walls", "screens", "furniture arrangement",
        "ceiling height", "sight lines", "privacy elements"
    ],
    "The Fire": [
        "fireplace", "gathering space", "seating arrangement",
        "focal point", "warm materials", "communal area"
    ],
    "House Cluster": [
        "grouped buildings", "shared spaces", "common areas", "pathways",
        "community layout", "connected homes", "public spaces"
    ],
    "Natural Doors and Windows": [
        "varied window sizes", "custom doorways", "organic openings",
        "natural light patterns", "view framing", "graduated windows"
    ],
    "Couple's Realm": [
        "private space", "separate area", "master suite", "quiet retreat",
        "personal sanctuary", "intimate space", "connected yet separate"
    ]
}

def compute_simple_saliency(image, output_base_path):
     # Step 0: Original Image
    cv2.imwrite(f"{output_base_path}_0_original.jpg", image)

    # Step 1: Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_base_path}_1_grayscale.jpg", gray)

    # Step 2: Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    cv2.imwrite(f"{output_base_path}_2_blurred.jpg", blurred)

    # Step 3: Saliency Map (difference)
    saliency_map = cv2.absdiff(gray, blurred)
    cv2.imwrite(f"{output_base_path}_3_saliency_map.jpg", saliency_map)

    # Step 4: Thresholded saliency (Otsu's method)
    _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_base_path}_4_thresholded.jpg", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Filter contours by size
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    return filtered_contours, thresh

    # centers = []
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 100:
    #         (x, y), radius = cv2.minEnclosingCircle(cnt)
    #         centers.append((int(x), int(y), int(radius)))

    # return centers


def get_chunked_embeddings(text, processor, model, chunk_size=60):
    """Process long text by breaking it into chunks and combining embeddings."""
    words = text.split()
    chunks = []

    # Create overlapping chunks of text
    for i in range(0, len(words), chunk_size//2):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Get embeddings for each chunk
    all_embeds = []
    for chunk in chunks:
        inputs = processor(
            text=[chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        embed = model.get_text_features(**inputs)
        all_embeds.append(embed)

    # Combine embeddings (average them)
    if all_embeds:
        combined_embed = torch.mean(torch.cat(all_embeds, dim=0), dim=0, keepdim=True)
        return combined_embed
    return None

def embed_patterns():
    """Embed all patterns, handling long descriptions properly."""
    all_pattern_embeds = []

    for pattern in patterns:
        pattern_embed = get_chunked_embeddings(pattern["description"], processor, model)
        if pattern_embed is not None:
            all_pattern_embeds.append(pattern_embed)

    # Combine all pattern embeddings into a single tensor
    return torch.cat(all_pattern_embeds, dim=0)

# Initialize text embeddings with full descriptions
text_embeds = embed_patterns()

def get_key_phrases(description, image_embed):
    """Extract and analyze key phrases from the pattern description."""
    # Split description into phrases
    phrases = [p.strip() for p in description.replace('.', ',').split(',')]
    phrases = [p for p in phrases if p]  # Remove empty phrases

    # Get similarity scores for each phrase
    phrase_results = []
    for phrase in phrases:
        # Get embedding for phrase
        phrase_embed = get_chunked_embeddings(phrase, processor, model)
        if phrase_embed is not None:
            # Calculate similarity
            similarity = torch.cosine_similarity(image_embed, phrase_embed)
            phrase_results.append({
                "text": phrase,
                "relevance": round(float(similarity[0]), 4)
            })

    # Sort by relevance and return top phrases
    phrase_results.sort(key=lambda x: x["relevance"], reverse=True)
    return phrase_results[:5]  # Return top 5 most relevant phrases

def generate_attention_map(image_tensor, text, model, processor, original_size):
    """Generate attention visualization for image regions relevant to the text."""
    # Process inputs with pattern-specific prompts
    text_inputs = processor(
        text=[f"Looking at this image, focus on {text}. Identify areas that show this pattern."],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    # Get attention weights from last layer
    with torch.no_grad():
        # Get vision features
        vision_outputs = model.vision_model(image_tensor)
        vision_last_hidden_state = vision_outputs.last_hidden_state  # [1, N, 768]

        # Calculate self-attention for vision features
        vision_features = vision_last_hidden_state[:, 1:, :]  # Remove [CLS] token

        # Calculate attention using dot product with mean feature vector
        mean_feature = vision_features.mean(dim=1, keepdim=True)  # [1, 1, 768]
        attention_scores = torch.matmul(vision_features, mean_feature.transpose(-2, -1))  # [1, N-1, 1]
        attention_scores = attention_scores.squeeze(-1)  # [1, N-1]

        # Reshape to patch grid
        patch_size = int(np.sqrt(vision_features.shape[1]))
        attention_map = attention_scores.reshape(-1, patch_size, patch_size)

        # Normalize attention weights
        attention_map = torch.nn.functional.softmax(attention_map.flatten(), dim=0).reshape(patch_size, patch_size)

        # Enhance contrast
        attention_map = torch.pow(attention_map, 2)  # Enhance strong activations
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Upscale to original image size
        attention_map = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return attention_map

def create_heatmap_overlay(image, attention_map, alpha=0.6, color_map='jet'):
    """Create a heatmap overlay with customizable parameters."""
    attention_np = attention_map.numpy()

    # Use different colormaps for different patterns
    heatmap = getattr(plt.cm, color_map)(attention_np)
    heatmap = Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))
    heatmap = heatmap.resize(image.size)

    # Enhance contrast for better visualization
    enhancer = ImageEnhance.Contrast(heatmap)
    heatmap = enhancer.enhance(1.5)

    return Image.blend(image, heatmap, alpha)

@app.post("/detect-patterns")
async def detect_patterns(image: UploadFile):
    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Process image
    inputs = processor(images=pil_img, return_tensors="pt")
    image_embed = model.get_image_features(**inputs)
    image_tensor = inputs.pixel_values

    similarity = torch.cosine_similarity(image_embed, text_embeds)
    top_k_results = similarity.topk(5)

    detailed_matches = []
    unique_id = uuid.uuid4()

    for score, idx in zip(top_k_results.values, top_k_results.indices):
        pattern = patterns[idx]


        # Get title similarity
        title_embed = get_chunked_embeddings(pattern["name"], processor, model)
        title_similarity = torch.cosine_similarity(image_embed, title_embed)

        try:
            # Create pattern-specific visualization parameters
            pattern_params = {
                "Window Place": {"color_map": "jet", "alpha": 0.6},
                "Thick Walls": {"color_map": "hot", "alpha": 0.7},
                "Positive Outdoor Space": {"color_map": "viridis", "alpha": 0.5},
                "Boundaries": {"color_map": "plasma", "alpha": 0.6},
                "Local Symmetry": {"color_map": "magma", "alpha": 0.5},
                "Indoor Sunlight": {"color_map": "autumn", "alpha": 0.65},
                "Site Repair": {"color_map": "YlOrRd", "alpha": 0.55},
                "Workspace Enclosure": {"color_map": "coolwarm", "alpha": 0.6},
                "The Fire": {"color_map": "inferno", "alpha": 0.7},
                "House Cluster": {"color_map": "RdYlBu", "alpha": 0.55},
                "Natural Doors and Windows": {"color_map": "viridis", "alpha": 0.6},
                "Couple's Realm": {"color_map": "plasma", "alpha": 0.5}
            }.get(pattern["name"], {"color_map": "jet", "alpha": 0.6})

           # Generate attention visualization with more specific pattern text
            pattern_text = (
                f"{pattern['name']}: {pattern['description']} "
                f"Look for: {', '.join(pattern_elements.get(pattern['name'], []))}"
            )
            attention_map = generate_attention_map(
                image_tensor,
                pattern_text,
                model,
                processor,
                pil_img.size
            )

            # Apply pattern-specific threshold
            threshold = attention_map.mean() + (attention_map.std() * 1.5)
            attention_map = torch.where(
                attention_map > threshold,
                attention_map,
                attention_map * 0.3  # Reduce non-significant areas
            )

            # Create unique filenames for each pattern
            pattern_slug = pattern['name'].lower().replace(' ', '_')
            heatmap_path = f"{unique_id}_{pattern_slug}_heatmap.jpg"

            # Create heatmap with pattern-specific parameters
            heatmap = create_heatmap_overlay(
                pil_img,
                attention_map,
                alpha=pattern_params["alpha"],
                color_map=pattern_params["color_map"]
            )
            heatmap.save(str(output_dir / heatmap_path))

            # Generate ROI visualization with enhanced attention map
            roi_path = generate_roi_visualization(
                pil_img,
                attention_map,
                pattern_slug,
                unique_id
            )

            visualizations = {
                "attention_map": f"/images/{heatmap_path}",
                "regions_of_interest": roi_path
            }
        except Exception as e:
            print(f"Visualization error for {pattern['name']}: {e}")
            visualizations = None

        # Add to detailed matches
        detailed_matches.append({
            "pattern_name": pattern["name"],
            "description": pattern["description"],
            "similarity_score": round(float(score), 4),
            "title_similarity": round(float(title_similarity[0]), 4),
            "explanation": {
                "key_phrases": get_key_phrases(pattern["description"], image_embed),
                "supporting_elements": analyze_supporting_elements(image_embed, pattern["name"]),
                "visualizations": visualizations
            }
        })

    return {
        "detailed_matches": detailed_matches,
        "analysis_metadata": {
            "image_size": pil_img.size,
            "confidence_threshold": 0.2
        }
    }

def generate_roi_visualization(image, attention_map, pattern_name, unique_id):
    """Generate visualization of specific regions of interest."""
    # Threshold attention map to find important regions
    threshold = attention_map.mean() + attention_map.std()
    roi_mask = attention_map > threshold

    # Find contours of important regions
    roi_mask_np = (roi_mask.numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        roi_mask_np,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw boxes around regions of interest
    image_with_roi = image.copy()
    draw = ImageDraw.Draw(image_with_roi)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline='red',
            width=2
        )

    # Save and return path
    roi_path = f"{unique_id}_{pattern_name.replace(' ', '_')}_roi.jpg"
    image_with_roi.save(str(output_dir / roi_path))
    return f"/images/{roi_path}"

def analyze_supporting_elements(image_embed, pattern_name):
    """Analyze supporting elements based on the pattern context."""

    # Define pattern-specific elements to check for


    # Get relevant elements for this pattern
    elements_to_check = pattern_elements.get(pattern_name, [])
    if not elements_to_check:
        return []

    # Check for presence of supporting elements
    element_inputs = processor(
        text=elements_to_check,
        return_tensors="pt",
        padding=True
    )
    element_embeds = model.get_text_features(**element_inputs)
    element_similarities = torch.cosine_similarity(image_embed, element_embeds)

    # Return elements that support the pattern
    supporting_elements = [
        {
            "element": elem,
            "confidence": round(float(score), 4),
            "contribution": "high" if score > 0.3 else "medium" if score > 0.2 else "low"
        }
        for elem, score in zip(elements_to_check, element_similarities)
        if score > 0.15  # Threshold for relevance
    ]

    return sorted(supporting_elements, key=lambda x: x["confidence"], reverse=True)

@app.post("/detect-centers")
async def detect_centers(image: UploadFile):
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    unique_id = uuid.uuid4()
    output_base_path = output_dir / f"{unique_id}"
    contours, saliency_map = compute_simple_saliency(img, str(output_base_path))

    annotated_img = img.copy()
    cv2.drawContours(annotated_img, contours, -1, (0, 255, 0), thickness=2)

    annotated_image_path = f"{output_base_path}_annotated.jpg"
    cv2.imwrite(annotated_image_path, annotated_img)

    return JSONResponse({
         "steps": {
            "original": f"/images/{unique_id}_0_original.jpg",
            "grayscale": f"/images/{unique_id}_1_grayscale.jpg",
            "blurred": f"/images/{unique_id}_2_blurred.jpg",
            "saliency_map": f"/images/{unique_id}_3_saliency_map.jpg",
            "thresholded": f"/images/{unique_id}_4_thresholded.jpg",
            "annotated": f"/images/{unique_id}_annotated.jpg"
        },
    })

