# Harmony Seeking

An experiment in spotting Christopher Alexander's *A Pattern Language* patterns (Window Place, Thick Walls, Indoor Sunlight…) in photos of spaces.

Upload a photo, and a Python service scores it against each pattern's description using CLIP. It returns the top matches with heatmaps, regions of interest and supporting elements.

## How it works

- **`ml-backend/`**: a FastAPI service using CLIP (`openai/clip-vit-base-patch32`), PyTorch, OpenCV and Pillow.
  - `POST /detect-patterns` compares the image embedding with pattern text embeddings.
  - `POST /detect-centers` shows a simple saliency pipeline step by step.
- **Laravel app**: handles the upload form, stores the image, calls the backend and renders the results.

## Run it

```bash
# ML backend (Python 3.12)
cd ml-backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p images
uvicorn main:app --port 8003

# Laravel app
composer install && npm install && npm run build
cp .env.example .env && php artisan key:generate
php artisan migrate && php artisan storage:link
php artisan serve
```

## Limitations

- The heatmaps come from the image's own patch features. They aren't conditioned on the pattern text, so each pattern gets the same map in a different colour.
- Scores are raw CLIP cosine similarities, not calibrated probabilities. Treat the rankings as a hint, not a verdict.
