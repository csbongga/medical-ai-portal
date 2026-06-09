import json
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import keras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# Claude API Service
from claude_service import claude_service

# ---- config ----
MODEL_PATH = "models/best_root_model.keras"
METADATA_PATH = "models/best_root_model.json"

# ---- global state loaded once at startup ----
state: dict = {}


def load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_idx_to_class(class_index: dict) -> dict:
    first_val = next(iter(class_index.values()))
    if isinstance(first_val, dict):
        return {
            str(k): {"code": v.get("code", str(k)), "name": v.get("name", "")}
            for k, v in class_index.items()
        }
    if isinstance(first_val, int):
        return {str(v): {"code": k, "name": k} for k, v in class_index.items()}
    return {str(k): {"code": str(v), "name": str(v)} for k, v in class_index.items()}


def preprocess(image_bytes: bytes, target_size: tuple) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # raw 0-255, EfficientNet preprocesses internally
    return np.expand_dims(arr, axis=0)       # (1, H, W, 3)


class MockModel:
    """Mock model for testing when real model file is not available"""
    def __init__(self):
        self.input_shape = (None, 224, 224, 3)

    def predict(self, x, verbose=0):
        # Mock prediction with realistic probabilities
        # Make IP (index 0) most likely for demo
        probs = np.random.dirichlet([3, 1, 1, 1, 1, 1, 1])  # 7 classes, IP weighted higher
        return [probs]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — load once
    print("Loading model...")

    # Try to load real model first
    model = None
    model_mode = "mock"

    if Path(MODEL_PATH).exists():
        try:
            model = keras.models.load_model(MODEL_PATH)
            model_mode = "real"
            print(f"SUCCESS: Real model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load real model: {e}")
            print("INFO: Falling back to mock model...")
    else:
        print(f"INFO: Model file not found: {MODEL_PATH}")
        print("INFO: Using mock model for demo...")

    if model is None:
        model = MockModel()

    input_shape = model.input_shape          # (None, H, W, C)
    target_size = (input_shape[2], input_shape[1])  # (W, H) for PIL

    print("Loading metadata...")
    try:
        metadata = load_metadata(METADATA_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        # Fallback metadata
        metadata = {
            "class_index": {
                "0": {"code": "IP", "name": "Irreversible pulpitis with Acute periodontitis"},
                "1": {"code": "IT", "name": "Impacted tooth (fully bony impaction)"},
                "2": {"code": "IR", "name": "Improper restoration with chronic apical periodontitis"},
                "3": {"code": "CAP", "name": "Chronic apical periodontitis with vertical bone loss"},
                "4": {"code": "ET", "name": "Embedded tooth"},
                "5": {"code": "DC", "name": "Dental caries (proximal)"},
                "6": {"code": "PD", "name": "Periodontitis"}
            }
        }

    idx_to_class = build_idx_to_class(metadata["class_index"])

    state["model"] = model
    state["target_size"] = target_size
    state["idx_to_class"] = idx_to_class
    state["input_shape"] = input_shape
    state["model_mode"] = model_mode
    print(f"Ready — mode: {model_mode}, input shape: {input_shape}, classes: {len(idx_to_class)}")
    yield
    state.clear()


app = FastAPI(
    title="Dental Root Disease Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        validation_result = claude_service.validate_dental_xray(image_bytes)
        return JSONResponse(validation_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image validation failed: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        x = preprocess(image_bytes, state["target_size"])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot process image: {e}")

    probs = state["model"].predict(x, verbose=0)[0]
    idx_to_class = state["idx_to_class"]

    predictions = sorted(
        [
            {
                "rank": 0,
                "index": int(i),
                "code": idx_to_class.get(str(i), {}).get("code", str(i)),
                "name": idx_to_class.get(str(i), {}).get("name", str(i)),
                "probability": round(float(probs[i]), 6),
                "probability_pct": round(float(probs[i]) * 100, 2),
            }
            for i in range(len(probs))
        ],
        key=lambda x: x["probability"],
        reverse=True,
    )
    for rank, item in enumerate(predictions, 1):
        item["rank"] = rank

    return JSONResponse({
        "filename": file.filename,
        "model_mode": state.get("model_mode", "unknown"),
        "top_prediction": {
            "code": predictions[0]["code"],
            "name": predictions[0]["name"],
            "probability_pct": predictions[0]["probability_pct"],
        },
        "all_predictions": predictions,
    })


# ---- Claude API Models ----
class TreatmentRequest(BaseModel):
    disease_name: str
    disease_code: str
    confidence: float
    patient_info: str = None

class ChatRequest(BaseModel):
    question: str
    conversation_history: List[Dict[str, str]] = []
    context: Dict[str, Any] = {}

class ApiResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = {}
    message: str = ""


# ---- Claude API Endpoints ----
@app.post("/treatment-advice")
async def get_treatment_advice(request: TreatmentRequest):
    """
    สร้างคำแนะนำการรักษาจากผลการทำนาย
    """
    try:
        recommendation = claude_service.generate_treatment_recommendation(
            disease_name=request.disease_name,
            disease_code=request.disease_code,
            confidence=request.confidence,
            patient_info=request.patient_info
        )

        return ApiResponse(
            success=True,
            data={
                "recommendation": recommendation,
                "disease_code": request.disease_code,
                "confidence": request.confidence
            },
            message="สร้างคำแนะนำการรักษาสำเร็จ"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดในการสร้างคำแนะนำ: {str(e)}"
        )


@app.post("/chat")
async def chat_with_claude(request: ChatRequest):
    """
    ตอบคำถามของแพทย์ในรูปแบบการสนทนา
    """
    try:
        response = claude_service.chat_with_doctor(
            question=request.question,
            conversation_history=request.conversation_history,
            context=request.context
        )

        return ApiResponse(
            success=True,
            data={
                "response": response,
                "question": request.question
            },
            message="ตอบคำถามสำเร็จ"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดในการตอบคำถาม: {str(e)}"
        )


@app.get("/claude-status")
def check_claude_status():
    """
    ตรวจสอบสถานะการเชื่อมต่อ Claude API
    """
    try:
        status = claude_service.test_connection()
        return status
    except Exception as e:
        return {
            "status": "error",
            "message": f"ไม่สามารถตรวจสอบสถานะได้: {str(e)}"
        }
