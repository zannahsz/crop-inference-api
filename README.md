## Crop image inference API (FastAPI)

### Run locally
1) Install Python 3.10+.
2) In this folder:
   - `pip install -r requirements.txt`
   - `uvicorn app:app --reload --port 8000`

### Test
- GET http://localhost:8000/health
- POST http://localhost:8000/predict (multipart form-data)
  - crop: lettuce | potato
  - file: image

Response:
{
  "crop": "lettuce",
  "prediction": "-N",
  "confidence": 0.93,
  "classes": [...]
}