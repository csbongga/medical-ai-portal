#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Medical AI Server
Combines Dental and Skin Diagnosis systems under a single port (9000)
"""
import os
import io
import json
import base64
from typing import List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import keras
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
import anthropic

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

# ---- Load Config ----
# Load environment variables from the dental app since it contains the ANTHROPIC_API_KEY
load_dotenv("medical-diagnosis-app/.env")

DENTAL_MODEL_PATH = "medical-diagnosis-app/models/best_root_model.keras"
DENTAL_METADATA_PATH = "medical-diagnosis-app/models/best_root_model.json"

SKIN_MODEL_PATH = "skin-diagnosis-app/models/best_model_final.keras"
SKIN_METADATA_PATH = "skin-diagnosis-app/models/skin_model_info.json"

# Global state for loaded systems
state: dict = {
    "dental": {},
    "skin": {}
}


# ---- ML Helper Functions ----
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
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


class MockModel:
    """Mock model for testing when real model file is not available"""
    def __init__(self, num_classes=7, is_dental=True):
        self.input_shape = (None, 224, 224, 3)
        self.num_classes = num_classes
        self.is_dental = is_dental

    def predict(self, x, verbose=0):
        # Make a realistic mock prediction distribution
        if self.is_dental:
            probs = np.random.dirichlet([3] + [1] * (self.num_classes - 1))
        else:
            probs = np.random.dirichlet([1, 1, 1, 1, 1, 4, 1])
        return [probs]


# ---- Combined Claude API Service ----
class UnifiedClaudeService:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("กรุณาตั้งค่า ANTHROPIC_API_KEY ใน .env file")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6"

        self.dental_system_prompt = """คุณเป็นระบบ AI ช่วยสนับสนุนการตัดสินใจของแพทย์ทันตกรรม โดยมีหน้าที่ให้คำแนะนำทางการแพทย์อ้างอิงหลักฐานทางวิทยาศาสตร์

🔶 บทบาทและข้อจำกัด:
- คุณเป็น "เครื่องมือช่วยตัดสินใจ" ไม่ใช่ "ตัวแทนแพทย์"
- การตัดสินใจสุดท้ายเป็นของแพทย์เสมอ
- ห้ามให้การวินิจฉัยขั้นสุดท้าย หรือฟันธงแทนแพทย์
- ให้คำแนะนำที่เป็นกลางและอ้างอิงหลักฐานทางการแพทย์

🔶 ข้อมูลที่ต้องพิจารณา:
- ผลการทำนายมาจากโมเดล AI ที่มีความแม่นยำประมาณ 90% (ไม่ใช่ 100%)
- ระดับ confidence ของการทำนายเป็นปัจจัยสำคัญที่ต้องแจ้งแพทย์
- ต้องแนะนำให้แพทย์ประเมินอาการทางคลินิกเพิ่มเติม

🔶 รูปแบบการตอบ:
- ตอบเป็นภาษาไทยเท่านั้น
- ใช้ภาษาทางการแพทย์ที่เหมาะสม
- ตอบเป็น JSON format เท่านั้น ห้ามเพิ่มข้อความอื่นใดๆ

🔶 JSON Structure ที่ต้องการ:
{
  "causes": "สาเหตุและข้อบ่งชี้ของโรค",
  "treatments": [
    {"type": "warning|primary|secondary", "number": "!", "label": "หัวข้อ", "description": "รายละเอียด"}
  ]
}

สำคัญมาก: ตอบเฉพาะ JSON object เท่านั้น ไม่ใส่ข้อความอื่นใดๆ เพิ่มเติม"""

        self.skin_system_prompt = """คุณเป็นระบบ AI ช่วยสนับสนุนการตัดสินใจของแพทย์ผิวหนัง โดยมีหน้าที่ให้คำแนะนำทางการแพทย์อ้างอิงหลักฐานทางวิทยาศาสตร์

🔶 บทบาทและข้อจำกัด:
- คุณเป็น "เครื่องมือช่วยตัดสินใจ" ไม่ใช่ "ตัวแทนแพทย์"
- การตัดสินใจสุดท้ายเป็นของแพทย์เสมอ
- ห้ามให้การวินิจฉัยขั้นสุดท้าย หรือฟันธงแทนแพทย์
- ให้คำแนะนำที่เป็นกลางและอ้างอิงหลักฐานทางการแพทย์

🔶 ข้อมูลที่ต้องพิจารณา:
- ผลการทำนายมาจากโมเดล AI ที่มีความแม่นยำประมาณ 90% (ไม่ใช่ 100%)
- ระดับ confidence ของการทำนายเป็นปัจจัยสำคัญที่ต้องแจ้งแพทย์
- ต้องแนะนำให้แพทย์ประเมินอาการทางคลินิกเพิ่มเติม

🔶 รูปแบบการตอบ:
- ตอบเป็นภาษาไทยเท่านั้น
- ใช้ภาษาทางการแพทย์ที่เหมาะสม
- ตอบเป็น JSON format เท่านั้น ห้ามเพิ่มข้อความอื่นใดๆ

🔶 JSON Structure ที่ต้องการ:
{
  "causes": "สาเหตุและข้อบ่งชี้ของโรคผิวหนัง",
  "treatments": [
    {"type": "warning|primary|secondary", "number": "!", "label": "หัวข้อ", "description": "รายละเอียด"}
  ]
}

สำคัญมาก: ตอบเฉพาะ JSON object เท่านั้น ไม่ใส่ข้อความอื่นใดๆ เพิ่มเติม"""

        self.dental_chat_prompt = """คุณเป็น AI ผู้ช่วยแพทย์ทันตกรรมที่เป็นมิตรและช่วยเหลือ โดยมีหน้าที่ตอบคำถามและให้คำแนะนำเบื้องต้น

🔶 บทบาทในการสนทนา:
- คุณเป็น AI ผู้ช่วยที่สามารถสนทนาได้อย่างเป็นธรรมชาติ
- ตอบคำถามเป็นภาษาไทยและใช้ภาษาที่เข้าใจง่าย
- หากเป็นคำถามทางการแพทย์ ให้คำแนะนำเบื้องต้นและแนะนำให้ปรึกษาแพทย์
- หากเป็นการทักทายหรือคำถามทั่วไป ให้ตอบแบบเป็นมิตร

🔶 ข้อจำกัด:
- ไม่ให้การวินิจฉัยขั้นสุดท้าย
- แนะนำให้ปรึกษาแพทย์สำหรับการรักษาที่เฉพาะเจาะจง
- ตอบเป็นข้อความปกติ ไม่ใช่ JSON format

🔶 ลักษณะการตอบ:
- ตอบเป็นภาษาไทย
- ใช้อีโมจิให้เหมาะสม
- เป็นมิตรและให้ความช่วยเหลือ
- หากมีข้อมูลบริบทการตรวจพบโรค สามารถอ้างอิงได้"""

        self.skin_chat_prompt = """คุณเป็น AI ผู้ช่วยแพทย์ผิวหนังที่เป็นมิตรและช่วยเหลือ โดยมีหน้าที่ตอบคำถามและให้คำแนะนำเบื้องต้น

🔶 บทบาทในการสนทนา:
- คุณเป็น AI ผู้ช่วยที่สามารถสนทนาได้อย่างเป็นธรรมชาติ
- ตอบคำถามเป็นภาษาไทยและใช้ภาษาที่เข้าใจง่าย
- หากเป็นคำถามทางการแพทย์ ให้คำแนะนำเบื้องต้นและแนะนำให้ปรึกษาแพทย์
- หากเป็นการทักทายหรือคำถามทั่วไป ให้ตอบแบบเป็นมิตร

🔶 ข้อจำกัด:
- ไม่ให้การวินิจฉัยขั้นสุดท้าย
- แนะนำให้ปรึกษาแพทย์สำหรับการรักษาที่เฉพาะเจาะจง
- ตอบเป็นข้อความปกติ ไม่ใช่ JSON format

🔶 ลักษณะการตอบ:
- ตอบเป็นภาษาไทย
- ใช้อีโมจิให้เหมาะสม
- เป็นมิตรและให้ความช่วยเหลือ
- หากมีข้อมูลบริบทการตรวจพบโรค สามารถอ้างอิงได้"""

    def generate_treatment_recommendation(
        self,
        disease_name: str,
        disease_code: str,
        confidence: float,
        patient_info: str = None,
        is_dental: bool = True
    ) -> str:
        system_prompt = self.dental_system_prompt if is_dental else self.skin_system_prompt
        example_cause = "การอักเสบของโพรงประสาทฟัน..." if is_dental else "รอยโรคร่วมกับความเสี่ยงผิวหนัง..."
        
        user_prompt = f"""โปรดวิเคราะห์และให้คำแนะนำการรักษาจากผลการทำนายต่อไปนี้:

📊 ผลการทำนายจากระบบ AI:
- โรคที่ตรวจพบ: {disease_name}
- รหัสโรค: {disease_code}
- ระดับความเชื่อมั่น: {confidence:.1f}%

"""
        if patient_info:
            user_prompt += f"\n👤 ข้อมูลผู้ป่วยเพิ่มเติม:\n{patient_info}\n"

        user_prompt += f"""
กรุณาตอบเป็น JSON format เท่านั้น ตามโครงสร้างที่กำหนด:

ตัวอย่าง JSON:
{{
  "causes": "{example_cause}",
  "treatments": [
    {{"type": "warning", "number": "!", "label": "ประเมินเบื้องต้น", "description": "ขั้นตอนการตรวจวินิจฉัยและสังเกตอาการเพิ่มเติม"}},
    {{"type": "primary", "number": "1", "label": "การรักษาหลัก", "description": "แนวทางการบำบัดรักษาหลัก"}},
    {{"type": "secondary", "number": "2", "label": "การรักษาเพิ่มเติม", "description": "การจ่ายยาหรือแนวทางเสริม"}},
    {{"type": "secondary", "number": "3", "label": "ติดตามผล", "description": "ระยะเวลานัดตรวจติดตามผล"}}
  ]
}}

สำคัญมาก:
- ตอบเป็น JSON object เท่านั้น ห้ามเพิ่มข้อความอื่น
- type ใช้ได้เฉพาะ: "warning", "primary", "secondary"
- ระดับความเชื่อมั่น: {confidence:.1f}%"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            response_text = response.content[0].text.strip()
            
            # Clean markdown JSON wrapping if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                json_response = json.loads(response_text)
                return json.dumps(json_response, ensure_ascii=False)
            except json.JSONDecodeError:
                return response_text
        except Exception as e:
            return json.dumps({"error": f"เกิดข้อผิดพลาดในการเชื่อมต่อ Claude API: {str(e)}"}, ensure_ascii=False)

    def chat_with_doctor(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        context: Dict[str, Any] = None,
        is_dental: bool = True
    ) -> str:
        system_prompt = self.dental_chat_prompt if is_dental else self.skin_chat_prompt
        messages = []

        if context:
            context_msg = "📋 ข้อมูลบริบทปัจจุบัน:\n"
            if context.get('disease'):
                context_msg += f"- โรคที่ตรวจพบ: {context['disease']}\n"
            if context.get('confidence'):
                context_msg += f"- ความเชื่อมั่น: {context['confidence']}\n"
            if context.get('patient_info'):
                context_msg += f"- ข้อมูลผู้ป่วย: {context['patient_info']}\n"

            messages.append({"role": "user", "content": context_msg})
            messages.append({"role": "assistant", "content": "รับทราบข้อมูลบริบทแล้วครับ มีอะไรให้ช่วยเหลือไหมครับ?"})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": question})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.4,
                system=system_prompt,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ Claude API: {str(e)}"

    def validate_image(self, image_bytes: bytes, is_dental: bool = True, media_type: str = "image/jpeg") -> Dict[str, Any]:
        """Validate if uploaded image is relevant to dental X-ray or skin lesion"""
        topic = "ภาพ X-ray ทางทันตกรรม (Dental X-ray)" if is_dental else "ภาพถ่ายรอยโรคผิวหนัง (Skin lesion)"
        target = "Dental X-ray" if is_dental else "Skin lesion"
        
        user_prompt = f"""โปรดวิเคราะห์ภาพนี้และตอบเป็น JSON format เท่านั้น:

ตรวจสอบว่าภาพนี้เป็น:
1. {topic} หรือไม่
2. มีโครงสร้างอวัยวะหรือรอยโรคที่สอดคล้องกับหัวข้อปรากฏหรือไม่
3. เป็นภาพทางการแพทย์ที่เกี่ยวข้องและเหมาะสมในการตรวจวินิจฉัยหรือไม่

ตอบเป็น JSON format เท่านั้น:
{{
  "is_valid": true/false,
  "confidence": 0-100,
  "message": "คำอธิบายภาษาไทย"
}}

หากเป็นภาพที่ถูกต้องตามหัวข้อ: is_valid = true
หากไม่ใช่: is_valid = false"""

        try:
            # Normalize media type for Claude API
            media_type = media_type.lower()
            if "png" in media_type:
                claude_media_type = "image/png"
            elif "webp" in media_type:
                claude_media_type = "image/webp"
            elif "gif" in media_type:
                claude_media_type = "image/gif"
            else:
                claude_media_type = "image/jpeg"

            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": claude_media_type,
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": user_prompt}
                ]
            }

            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[message]
            )

            response_text = response.content[0].text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                result = json.loads(response_text)
                return {
                    "is_valid": result.get("is_valid", False),
                    "confidence": result.get("confidence", 0),
                    "message": result.get("message", "ไม่สามารถวิเคราะห์ได้")
                }
            except json.JSONDecodeError:
                return {
                    "is_valid": False,
                    "confidence": 0,
                    "message": f"ไม่สามารถวิเคราะห์ภาพได้ กรุณาอัปโหลดภาพที่เหมาะสมในการตรวจวินิจฉัย {target}"
                }
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0,
                "message": f"เกิดข้อผิดพลาดในการตรวจสอบภาพ: {str(e)}"
            }

    def test_connection(self) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": "สวัสดี"}]
            )
            return {"status": "success", "message": "เชื่อมต่อสำเร็จ", "response": response.content[0].text}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ---- Global API Service Instance ----
claude_service = UnifiedClaudeService()


# ---- Startup / Shutdown Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup Dental System
    print("Loading Dental ML Model...")
    dental_model = None
    dental_mode = "mock"
    if Path(DENTAL_MODEL_PATH).exists():
        try:
            dental_model = keras.models.load_model(DENTAL_MODEL_PATH)
            dental_mode = "real"
            print(f"SUCCESS: Dental model loaded: {DENTAL_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load dental model: {e}")
    
    if dental_model is None:
        print("INFO: Falling back to Dental mock model...")
        dental_model = MockModel(num_classes=7, is_dental=True)

    try:
        dental_meta = load_metadata(DENTAL_METADATA_PATH)
        dental_idx_to_class = build_idx_to_class(dental_meta["class_index"])
    except Exception as e:
        print(f"ERROR: Failed to load dental metadata: {e}")
        # Default fallback metadata
        dental_idx_to_class = {
            "0": {"code": "IP", "name": "Irreversible pulpitis with Acute periodontitis"},
            "1": {"code": "IT", "name": "Impacted tooth (fully bony impaction)"},
            "2": {"code": "IR", "name": "Improper restoration with chronic apical periodontitis"},
            "3": {"code": "CAP", "name": "Chronic apical periodontitis with vertical bone loss"},
            "4": {"code": "ET", "name": "Embedded tooth"},
            "5": {"code": "DC", "name": "Dental caries (proximal)"},
            "6": {"code": "PD", "name": "Periodontitis"}
        }

    state["dental"] = {
        "model": dental_model,
        "target_size": (dental_model.input_shape[2], dental_model.input_shape[1]),
        "idx_to_class": dental_idx_to_class,
        "model_mode": dental_mode
    }

    # Setup Skin System
    print("Loading Skin ML Model...")
    skin_model = None
    skin_mode = "mock"
    if Path(SKIN_MODEL_PATH).exists():
        try:
            skin_model = keras.models.load_model(SKIN_MODEL_PATH)
            skin_mode = "real"
            print(f"SUCCESS: Skin model loaded: {SKIN_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load skin model: {e}")
    
    if skin_model is None:
        print("INFO: Falling back to Skin mock model...")
        skin_model = MockModel(num_classes=7, is_dental=False)

    try:
        skin_meta = load_metadata(SKIN_METADATA_PATH)
        if "class_index" in skin_meta:
            skin_idx_to_class = build_idx_to_class(skin_meta["class_index"])
        else:
            # handle 'classes' key format
            class_index = {}
            for i, class_name in enumerate(skin_meta["classes"]):
                full_name = skin_meta.get("class_full_names", {}).get(class_name, class_name)
                class_index[str(i)] = {"code": class_name.upper(), "name": full_name}
            skin_idx_to_class = build_idx_to_class(class_index)
    except Exception as e:
        print(f"ERROR: Failed to load skin metadata: {e}")
        # Default fallback metadata
        skin_idx_to_class = {
            "0": {"code": "AKIEC", "name": "Actinic keratoses / รอยโรคก่อนมะเร็ง"},
            "1": {"code": "BCC", "name": "Basal cell carcinoma / มะเร็งเซลล์ฐาน"},
            "2": {"code": "BKL", "name": "Benign keratosis / กระเนื้องอกชนิดไม่ร้าย"},
            "3": {"code": "DF", "name": "Dermatofibroma / เนื้องอกผิวหนังชนิดไม่ร้าย"},
            "4": {"code": "MEL", "name": "Melanoma / มะเร็งผิวหนังเมลาโนมา"},
            "5": {"code": "NV", "name": "Melanocytic nevi / ไฝปกติ"},
            "6": {"code": "VASC", "name": "Vascular lesions / รอยโรคหลอดเลือด"}
        }

    state["skin"] = {
        "model": skin_model,
        "target_size": (skin_model.input_shape[2], skin_model.input_shape[1]),
        "idx_to_class": skin_idx_to_class,
        "model_mode": skin_mode
    }

    print("\n" + "="*60)
    print("UNIFIED MEDICAL AI SERVER STARTED")
    print("="*60)
    print("Unified Portal: http://localhost:9000/")
    print("Dental System:  http://localhost:9000/dental/")
    print("Skin System:    http://localhost:9000/skin/")
    print("="*60 + "\n")
    yield
    state["dental"].clear()
    state["skin"].clear()


app = FastAPI(
    title="Unified Medical AI Portal Server",
    description="Unified API server hosting Dental and Skin diagnostic systems.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Portal / Landing Routes ----
@app.get("/", response_class=HTMLResponse)
async def get_portal():
    portal_html_path = Path("medical-ai-portal/index.html")
    if portal_html_path.exists():
        with open(portal_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Update portal link redirection to use unified server endpoints
        content = content.replace("http://localhost:8000", "/dental/")
        content = content.replace("http://localhost:8001", "/skin/")
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>AI Medical Portal</h1><p>Portal file not found.</p>", status_code=404)


# ---- Dental Frontend ----
@app.get("/dental/", response_class=HTMLResponse)
async def get_dental_page():
    dental_html_path = Path("medical-diagnosis-app/static/index.html")
    if dental_html_path.exists():
        with open(dental_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Dental System</h1><p>Static index.html not found.</p>", status_code=404)


# ---- Skin Frontend ----
@app.get("/skin/", response_class=HTMLResponse)
async def get_skin_page():
    skin_html_path = Path("skin-diagnosis-app/static/index.html")
    if skin_html_path.exists():
        with open(skin_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Skin System</h1><p>Static index.html not found.</p>", status_code=404)


# ---- Generic Model Prediction Handler ----
def handle_prediction(image_bytes: bytes, sys_key: str):
    sys_state = state[sys_key]
    model = sys_state["model"]
    target_size = sys_state["target_size"]
    idx_to_class = sys_state["idx_to_class"]
    
    try:
        x = preprocess(image_bytes, target_size)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot process image: {e}")

    probs = model.predict(x, verbose=0)[0]
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
        key=lambda val: val["probability"],
        reverse=True,
    )
    for rank, item in enumerate(predictions, 1):
        item["rank"] = rank

    return {
        "model_mode": sys_state["model_mode"],
        "top_prediction": {
            "code": predictions[0]["code"],
            "name": predictions[0]["name"],
            "probability_pct": predictions[0]["probability_pct"],
        },
        "all_predictions": predictions,
    }


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


# ==========================================
# 🦷 DENTAL APIS (mounted under /dental/*)
# ==========================================

@app.get("/dental/health")
async def dental_health():
    return {"status": "ok", "mode": state["dental"]["model_mode"]}


@app.post("/dental/validate-image")
async def dental_validate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    validation_result = claude_service.validate_image(image_bytes, is_dental=True, media_type=file.content_type)
    return JSONResponse(validation_result)


@app.post("/dental/predict")
async def dental_predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    res = handle_prediction(image_bytes, "dental")
    res["filename"] = file.filename
    return JSONResponse(res)


@app.post("/dental/treatment-advice")
async def dental_treatment_advice(request: TreatmentRequest):
    try:
        rec = claude_service.generate_treatment_recommendation(
            disease_name=request.disease_name,
            disease_code=request.disease_code,
            confidence=request.confidence,
            patient_info=request.patient_info,
            is_dental=True
        )
        # Parse recommendation if it's a string containing JSON
        try:
            rec_parsed = json.loads(rec)
        except json.JSONDecodeError:
            rec_parsed = rec

        return ApiResponse(
            success=True,
            data={
                "recommendation": rec_parsed,
                "disease_code": request.disease_code,
                "confidence": request.confidence
            },
            message="สร้างคำแนะนำการรักษาสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dental/chat")
async def dental_chat(request: ChatRequest):
    try:
        resp = claude_service.chat_with_doctor(
            question=request.question,
            conversation_history=request.conversation_history,
            context=request.context,
            is_dental=True
        )
        return ApiResponse(
            success=True,
            data={"response": resp, "question": request.question},
            message="ตอบคำถามสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dental/claude-status")
def dental_claude_status():
    return claude_service.test_connection()


# ==========================================
# 🔬 SKIN APIS (mounted under /skin/*)
# ==========================================

@app.get("/skin/health")
async def skin_health():
    return {"status": "ok", "mode": state["skin"]["model_mode"]}


@app.post("/skin/validate-image")
async def skin_validate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    validation_result = claude_service.validate_image(image_bytes, is_dental=False, media_type=file.content_type)
    return JSONResponse(validation_result)


@app.post("/skin/predict")
async def skin_predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    res = handle_prediction(image_bytes, "skin")
    res["filename"] = file.filename
    return JSONResponse(res)


@app.post("/skin/treatment-advice")
async def skin_treatment_advice(request: TreatmentRequest):
    try:
        rec = claude_service.generate_treatment_recommendation(
            disease_name=request.disease_name,
            disease_code=request.disease_code,
            confidence=request.confidence,
            patient_info=request.patient_info,
            is_dental=False
        )
        # Parse recommendation if it's a string containing JSON
        try:
            rec_parsed = json.loads(rec)
        except json.JSONDecodeError:
            rec_parsed = rec

        return ApiResponse(
            success=True,
            data={
                "recommendation": rec_parsed,
                "disease_code": request.disease_code,
                "confidence": request.confidence
            },
            message="สร้างคำแนะนำการรักษาสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skin/chat")
async def skin_chat(request: ChatRequest):
    try:
        resp = claude_service.chat_with_doctor(
            question=request.question,
            conversation_history=request.conversation_history,
            context=request.context,
            is_dental=False
        )
        return ApiResponse(
            success=True,
            data={"response": resp, "question": request.question},
            message="ตอบคำถามสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skin/claude-status")
def skin_claude_status():
    return claude_service.test_connection()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
