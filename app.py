from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI()

# allow Next.js dev server to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return model, classes, tfm

MODELS = {
    "lettuce": load_model("lettuce_efficientnetb0.pt"),
    "potato":  load_model("potato_efficientnetb0.pt"),
}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
@torch.no_grad()
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    crop = crop.lower().strip()
    if crop not in MODELS:
        return {"error": "crop must be 'lettuce' or 'potato'"}

    model, classes, tfm = MODELS[crop]

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = tfm(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())

    return {
        "crop": crop,
        "prediction": classes[idx],
        "confidence": float(probs[idx]),
        "classes": classes,  # helpful for UI display
    }