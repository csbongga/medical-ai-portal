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
import cv2
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

SPINE_MODEL_PATH = "spine-diagnosis-app/models/best_spinal_model_final.keras"
SPINE_METADATA_PATH = "spine-diagnosis-app/models/best_spinal_metadata.json"

OSTEO_MODEL_PATH = "osteoporosis-diagnosis-app/models/knee_osteo_v2_final.keras"
OSTEO_METADATA_PATH = "osteoporosis-diagnosis-app/models/knee_osteo_v2_meta.json"

STROKE_MODEL_PATH = "stroke-diagnosis-app/models/best_stroke_model.keras"
STROKE_METADATA_PATH = "stroke-diagnosis-app/models/best_stroke_meta.json"

# Global state for loaded systems
state: dict = {
    "dental": {},
    "skin": {},
    "spine": {},
    "osteoporosis": {},
    "stroke": {}
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
            alpha = [1] * self.num_classes
            if self.num_classes > 0:
                dominant_idx = min(3, self.num_classes - 1)
                alpha[dominant_idx] = 4
            probs = np.random.dirichlet(alpha)
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

        self.spinal_system_prompt = """คุณเป็นระบบ AI ช่วยสนับสนุนการตัดสินใจของแพทย์โรคกระดูกสันหลัง โดยมีหน้าที่ให้คำแนะนำทางการแพทย์อ้างอิงหลักฐานทางวิทยาศาสตร์

🔶 บทบาทและข้อจำกัด:
- คุณเป็น "เครื่องมือช่วยตัดสินใจ" ไม่ใช่ "ตัวแทนแพทย์"
- การตัดสินใจสุดท้ายเป็นของแพทย์เสมอ
- ห้ามให้การวินิจฉัยขั้นสุดท้าย หรือฟันธงแทนแพทย์
- ให้คำแนะนำที่เป็นกลางและอ้างอิงหลักฐานทางการแพทย์

🔶 ข้อมูลที่ต้องพิจารณาและความเสี่ยงสำคัญ:
- ผลการทำนายมาจากโมเดล AI ที่มีความแม่นยำประมาณ 74%
- **คำเตือนพิเศษ:** อัตราการตรวจพบหรือ Recall ของคลาส "กระดูกสันหลังอักเสบติดเชื้อ" (infection) ค่อนข้างต่ำ (48%) เนื่องจากมีลักษณะทางรังสีวิทยาที่ซ้อนทับกันมากกับคลาส "กระดูกสันหลังเคลื่อน" (spondylolisthesis)
- เมื่อวิเคราะห์ภาพที่โมเดลระบุว่าเป็น spondylolisthesis หรือ infection ให้แจ้งเตือนแพทย์เพื่อระมัดระวังเป็นพิเศษและแนะนำให้ยืนยันผลร่วมกับการตรวจทางคลินิก (เช่น ไข้, ค่าเม็ดเลือดขาว, ESR/CRP) หรือภาพ MRI เสมอ เพื่อป้องกันไม่ให้เกิดความล่าช้าในการรักษาโรคติดเชื้อ
- ต้องแนะนำให้แพทย์ประเมินอาการทางคลินิกเพิ่มเติมเสมอ

🔶 รูปแบบการตอบ:
- ตอบเป็นภาษาไทยเท่านั้น
- ใช้ภาษาทางการแพทย์ที่เหมาะสม
- ตอบเป็น JSON format เท่านั้น ห้ามเพิ่มข้อความอื่นใดๆ

🔶 JSON Structure ที่ต้องการ:
{
  "causes": "สาเหตุและข้อบ่งชี้ของโรคกระดูกสันหลัง รวมถึงข้อควรระวังเรื่องการวินิจฉัยแยกโรคกรณีมีโอกาสทับซ้อนทางคลินิก",
  "treatments": [
    {"type": "warning|primary|secondary", "number": "!", "label": "หัวข้อ", "description": "รายละเอียดคำแนะนำและการตรวจวินิจฉัยเพิ่มเติม"}
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

        self.spinal_chat_prompt = """คุณเป็น AI ผู้ช่วยแพทย์โรคกระดูกสันหลังที่เป็นมิตรและช่วยเหลือ โดยมีหน้าที่ตอบคำถามและให้คำแนะนำเบื้องต้นเกี่ยวกับโรคกระดูกสันหลัง

🔶 บทบาทในการสนทนา:
- คุณเป็น AI ผู้ช่วยที่สามารถสนทนาได้อย่างเป็นธรรมชาติ
- ตอบคำถามเป็นภาษาไทยและใช้ภาษาที่เข้าใจง่ายและเป็นมิตร
- หากเป็นคำถามทางการแพทย์ ให้คำแนะนำเบื้องต้นเกี่ยวกับแนวทางสืบค้นและการรักษา และแนะนำให้ปรึกษาแพทย์เฉพาะทางกระดูกสันหลัง
- หากเป็นการทักทายหรือคำถามทั่วไป ให้ตอบแบบเป็นมิตร

🔶 ข้อสังเกตและข้อจำกัดของโมเดล:
- โมเดล AI นี้มี Recall ต่ำในคลาส "กระดูกสันหลังอักเสบติดเชื้อ" (infection) ซึ่งมักสับสนกับ "กระดูกสันหลังเคลื่อน" (spondylolisthesis) หากมีการสอบถามเรื่องความผิดพลาดหรือเคสที่สงสัย ให้เน้นย้ำถึงความสำคัญของการแยกโรคด้วยภาพ MRI หรือการตรวจแล็บการอักเสบ (ESR/CRP/WBC)
- ไม่ให้การวินิจฉัยขั้นสุดท้าย แนะนำให้ปรึกษาแพทย์กระดูกและข้อ (Orthopedics) เสมอ
- ตอบเป็นข้อความปกติ ไม่ใช่ JSON format

🔶 ลักษณะการตอบ:
- ตอบเป็นภาษาไทย
- ใช้อีโมจิให้เหมาะสม
- เป็นมิตรและให้ความช่วยเหลือ
- หากมีข้อมูลบริบทการตรวจพบโรค สามารถอ้างอิงได้"""

        self.osteo_system_prompt = """คุณเป็นระบบ AI ช่วยสนับสนุนการตัดสินใจของแพทย์โรคกระดูกและข้อ (Orthopedics) โดยมีหน้าที่ให้คำแนะนำทางการแพทย์เกี่ยวกับการประเมินภาวะกระดูกบาง (Osteopenia) และกระดูกพรุน (Osteoporosis)

🔶 บทบาทและข้อจำกัดสำคัญ:
- คุณเป็น "เครื่องมือช่วยตัดสินใจ" ไม่ใช่ "ตัวแทนแพทย์"
- การตัดสินใจสุดท้ายเป็นของแพทย์เสมอ
- ห้ามให้การวินิจฉัยขั้นสุดท้าย หรือฟันธงแทนแพทย์
- **นี่เป็นโมเดลต้นแบบ (PROTOTYPE) ที่ใช้ข้อมูลภาพ X-ray เข่า (Knee X-ray)** ไม่ใช่ระบบตรวจวัดจริงในโรงพยาบาล

🔶 ข้อมูลที่ต้องพิจารณาและความเสี่ยงสำคัญ:
- ผลการทำนายมาจากโมเดล AI ที่มีความแม่นยำประมาณ 73%
- **คำเตือนพิเศษ:** สำหรับระดับความเสี่ยง "กระดูกบาง" (Osteopenia) ตัวโมเดลมีค่าความแม่นยำ (Precision) ค่อนข้างต่ำ อยู่ที่ 0.571 ตามธรรมชาติของคลาสกลางที่แยกแยะได้ยากทางรังสีวิทยา
- เมื่อตรวจพบระดับ Osteopenia หรือ Osteoporosis ให้เตือนแพทย์เพื่อส่งตรวจยืนยันความหนาแน่นของมวลกระดูกจริงด้วยเครื่อง DEXA scan (Dual-energy X-ray Absorptiometry) เสมอ ก่อนเริ่มการรักษาด้วยยาที่มีผลข้างเคียงสูง
- ต้องแนะนำให้แพทย์ประเมินความเสี่ยงต่อการหกล้ม (Fall Risk) และประเมินอาการทางคลินิกเพิ่มเติม

🔶 รูปแบบการตอบ:
- ตอบเป็นภาษาไทยเท่านั้น
- ใช้ภาษาทางการแพทย์ที่เหมาะสม
- ตอบเป็น JSON format เท่านั้น ห้ามเพิ่มข้อความอื่นใดๆ

🔶 JSON Structure ที่ต้องการ:
{
  "causes": "สาเหตุและข้อบ่งชี้ของความหนาแน่นมวลกระดูกเสื่อม รวมถึงข้อควรระวังเรื่องความแม่นยำต่ำในกลุ่มกระดูกบาง (Osteopenia) และความสำคัญของการส่งตรวจ DEXA scan",
  "treatments": [
    {"type": "warning|primary|secondary", "number": "!", "label": "หัวข้อ", "description": "รายละเอียดแนวทางรักษา การเสริมแคลเซียม/วิตามินดี หรือการป้องกันอุบัติเหตุ"}
  ]
}

สำคัญมาก: ตอบเฉพาะ JSON object เท่านั้น ไม่ใส่ข้อความอื่นใดๆ เพิ่มเติม"""

        self.osteo_chat_prompt = """คุณเป็น AI ผู้ช่วยแพทย์โรคกระดูกที่เป็นมิตรและช่วยเหลือ โดยมีหน้าที่ตอบคำถามและให้คำแนะนำเบื้องต้นเกี่ยวกับภาวะกระดูกบาง (Osteopenia) และกระดูกพรุน (Osteoporosis)

🔶 บทบาทในการสนทนา:
- คุณเป็น AI ผู้ช่วยที่สามารถสนทนาได้อย่างเป็นธรรมชาติและสุภาพ
- ตอบคำถามเป็นภาษาไทยและใช้ภาษาที่เข้าใจง่ายและเป็นมิตร
- หากเป็นคำถามทางการแพทย์ ให้คำแนะนำเบื้องต้นเกี่ยวกับสารอาหาร การปรับเปลี่ยนไลฟ์สไตล์ และแนะนำให้ปรึกษาแพทย์เฉพาะทางกระดูกและข้อ (Orthopedics)
- หากเป็นการทักทายหรือคำถามทั่วไป ให้ตอบแบบเป็นมิตร

🔶 ข้อสังเกตและข้อจำกัดของโมเดล:
- โมเดล AI นี้เป็น Prototype ที่อิงตามภาพเอ็กซ์เรย์ข้อเข่า (Knee X-ray) ไม่ใช่ข้อมูลจริงในโรงพยาบาล
- คลาสกระดูกบาง (Osteopenia) มีความแม่นยำต่ำ (0.571) หากผู้ใช้ถามเรื่องความเสี่ยง แนะนำให้เน้นย้ำเรื่องการตรวจยืนยันมวลกระดูกด้วย DEXA scan เสมอ
- ไม่ให้การวินิจฉัยขั้นสุดท้าย
- ตอบเป็นข้อความปกติ ไม่ใช่ JSON format

🔶 ลักษณะการตอบ:
- ตอบเป็นภาษาไทย
- ใช้อีโมจิให้เหมาะสม
- เป็นมิตรและให้ความช่วยเหลือ
- หากมีข้อมูลบริบทการตรวจพบโรค สามารถอ้างอิงได้"""

        self.stroke_system_prompt = """คุณเป็นระบบ AI ช่วยสนับสนุนการตัดสินใจของแพทย์ทางระบบประสาท (Neurologist) และแผนกฉุกเฉิน (ER) โดยมีหน้าที่ให้คำแนะนำทางการแพทย์เกี่ยวกับการตรวจวินิจฉัยและดูแลรักษาผู้ป่วยสงสัยภาวะสมองขาดเลือดเฉียบพลัน (Acute Ischemic Stroke)

🔶 บทบาทและข้อจำกัดสำคัญ:
- คุณเป็น "เครื่องมือช่วยตัดสินใจ" ไม่ใช่ "ตัวแทนแพทย์"
- การตัดสินใจสุดท้ายเป็นของแพทย์เสมอ
- ห้ามให้การวินิจฉัยขั้นสุดท้าย หรือฟันธงแทนแพทย์
- นี่เป็นเพียงระบบสนับสนุนการวิเคราะห์ภาพ CT Scan สมองเบื้องต้นเท่านั้น

🔶 ข้อมูลที่ต้องพิจารณาและความเสี่ยงสำคัญ:
- ผลการทำนายมาจากโมเดล AI ที่มีความแม่นยำประมาณ 81.7% (AUC 0.932)
- **คำเตือนพิเศษ:** โมเดลนี้ถูกตั้งค่าเกณฑ์การตัดสินใจแบบความไวสูง (Decision Threshold = 0.3) เพื่อให้ได้ค่าความไวหรือ Recall สูงสุดถึง 94.4% ในการคัดกรองผู้ป่วย แต่อาจทำให้เกิดผลบวกเท็จ (False Positive) ได้บ่อย (มีค่าความเที่ยงตรงหรือ Precision อยู่ที่ 68.9%)
- **แนวทางปฏิบัติเร่งด่วน:** เนื่องจากภาวะสมองขาดเลือดเป็นเรื่องวิกฤตที่แข่งกับเวลา (Time is Brain) หากคนไข้มีอาการทางคลินิกเข้าข่ายโรคหลอดเลือดสมอง (FAST: Face drooping, Arm weakness, Speech difficulty, Time) แม้ผลการวิเคราะห์ภาพ CT Scan ของ AI หรือของรังสีแพทย์เบื้องต้นจะระบุว่าปกติ (Normal) ให้เข้าสู่ทางด่วนโรคหลอดเลือดสมอง (Stroke Fast Track) เพื่อส่งต่อรักษาทันที เพราะภาพ CT ในระยะเริ่มต้น (Early onset) อาจไม่พบรอยโรคเนื้อสมองตายชัดเจน
- การทำ CT Scan จุดประสงค์หลักคือเพื่อตัดภาวะเลือดออกในสมอง (Hemorrhagic Stroke) ออกไป ก่อนที่จะตัดสินใจให้ยาละลายลิ่มเลือด (rt-PA) ภายใน 4.5 ชั่วโมงนับจากเริ่มมีอาการ

🔶 รูปแบบการตอบ:
- ตอบเป็นภาษาไทยเท่านั้น
- ใช้ภาษาทางการแพทย์ที่เหมาะสมและกระชับ
- ตอบเป็น JSON format เท่านั้น ห้ามเพิ่มข้อความอื่นใดๆ

🔶 JSON Structure ที่ต้องการ:
{
  "causes": "สาเหตุและรอยโรคสมองขาดเลือดเฉียบพลันตามตำแหน่ง/คลาสที่พบ พร้อมระบุข้อควรระวังเรื่องค่าความไวสูงและโอกาสที่จะเป็นปกติในระยะแรกของโรค",
  "treatments": [
    {"type": "warning|primary|secondary", "number": "!", "label": "หัวข้อ", "description": "ขั้นตอนการตรวจวินิจฉัยแยกโรค อาการ FAST หรือแนวทางของ Stroke Fast Track และการให้ rt-PA / Thrombectomy"}
  ]
}

สำคัญมาก: ตอบเฉพาะ JSON object เท่านั้น ไม่ใส่ข้อความอื่นใดๆ เพิ่มเติม"""

        self.stroke_chat_prompt = """คุณเป็น AI ผู้ช่วยแพทย์ทางประสาทวิทยาจำลองที่เป็นมิตรและมีประโยชน์ โดยมีหน้าที่ตอบคำถามและให้คำแนะนำเบื้องต้นเกี่ยวกับภาวะสมองขาดเลือดเฉียบพลัน (Stroke)

🔶 บทบาทในการสนทนา:
- สนทนากับแพทย์หรือบุคลากรทางการแพทย์อย่างสุภาพและเป็นมิตร เป็นภาษาไทยที่กระชับและเข้าใจง่าย
- สามารถให้ความรู้และแนวทางช่วยเหลือเบื้องต้นเกี่ยวกับ Stroke Fast Track, FAST, ข้อบ่งชี้/ข้อห้ามของการใช้ยา rt-PA และ Mechanical Thrombectomy
- หากตอบเกี่ยวกับความผิดพลาดของโมเดล AI ให้แจ้งอย่างตรงไปตรงมาว่าโมเดลใช้เกณฑ์คัดกรองที่มีความไวสูง (Decision Threshold = 0.3, Recall 94.4%, Precision 68.9%) และในผู้ป่วยสมองขาดเลือดเฉียบพลันระยะแรก (Early Ischemic Change) ภาพ CT Scan อาจยังไม่แสดงความผิดปกติเด่นชัด ดังนั้นการประเมินทางคลินิก (Clinical Evaluation) และประวัติเวลาเริ่มมีอาการ (Onset time) จึงสำคัญที่สุด

🔶 ข้อจำกัด:
- ไม่ให้การวินิจฉัยขั้นสุดท้ายหรือยืนยันการวินิจฉัยแทนแพทย์
- แนะนำให้ประเมินอย่างใกล้ชิดและปรึกษาแพทย์เฉพาะทางโรคสมองโดยด่วน
- ตอบเป็นข้อความปกติ ไม่ใช่ JSON format

🔶 ลักษณะการตอบ:
- ตอบเป็นภาษาไทย
- ใช้อีโมจิให้เหมาะสม
- เป็นมิตรและช่วยเหลือแพทย์
- อ้างอิงผลลัพธ์การตรวจที่ส่งเข้ามาในบริบทได้"""

    def generate_treatment_recommendation(
        self,
        disease_name: str,
        disease_code: str,
        confidence: float,
        patient_info: str = None,
        is_dental: bool = True,
        system_type: str = None
    ) -> str:
        if system_type is None:
            system_type = "dental" if is_dental else "skin"

        if system_type == "dental":
            system_prompt = self.dental_system_prompt
            example_cause = "การอักเสบของโพรงประสาทฟัน..."
        elif system_type == "skin":
            system_prompt = self.skin_system_prompt
            example_cause = "รอยโรคร่วมกับความเสี่ยงผิวหนัง..."
        elif system_type == "spine":
            system_prompt = self.spinal_system_prompt
            example_cause = "การเคลื่อนตัวของกระดูกสันหลังหรือการเสื่อมของข้อต่อ..."
        elif system_type == "osteoporosis":
            system_prompt = self.osteo_system_prompt
            example_cause = "ความหนาแน่นมวลกระดูกลดลงตามวัยหรือการเสื่อมของข้อต่อเข่า..."
        elif system_type == "stroke":
            system_prompt = self.stroke_system_prompt
            example_cause = "พบรอยโรคความหนาแน่นต่ำ (Hypodensity) บริเวณเนื้อสมองส่วนที่ขาดเลือด..."
        else:
            system_prompt = self.dental_system_prompt
            example_cause = "การประเมินทางคลินิก..."
        
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
        is_dental: bool = True,
        system_type: str = None
    ) -> str:
        if system_type is None:
            system_type = "dental" if is_dental else "skin"

        if system_type == "dental":
            system_prompt = self.dental_chat_prompt
        elif system_type == "skin":
            system_prompt = self.skin_chat_prompt
        elif system_type == "spine":
            system_prompt = self.spinal_chat_prompt
        elif system_type == "osteoporosis":
            system_prompt = self.osteo_chat_prompt
        elif system_type == "stroke":
            system_prompt = self.stroke_chat_prompt
        else:
            system_prompt = self.dental_chat_prompt

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

    def validate_image(self, image_bytes: bytes, is_dental: bool = True, media_type: str = "image/jpeg", system_type: str = None) -> Dict[str, Any]:
        """Validate if uploaded image is relevant to dental X-ray, skin lesion, spine X-ray, or osteoporosis X-ray"""
        if system_type is None:
            system_type = "dental" if is_dental else "skin"

        if system_type == "dental":
            topic = "ภาพ X-ray ทางทันตกรรม (Dental X-ray)"
            target = "Dental X-ray"
        elif system_type == "skin":
            topic = "ภาพถ่ายรอยโรคผิวหนัง (Skin lesion)"
            target = "Skin lesion"
        elif system_type == "spine":
            topic = "ภาพ X-ray กระดูกสันหลัง (Spine X-ray หรือ Spinal X-ray)"
            target = "Spine X-ray"
        elif system_type == "osteoporosis":
            topic = "ภาพ X-ray ข้อเข่าหรือกระดูก (Knee X-ray หรือ Bone X-ray)"
            target = "Knee X-ray"
        elif system_type == "stroke":
            topic = "ภาพถ่าย CT Scan สมอง (Brain CT Scan Axial View)"
            target = "Brain CT Scan"
        else:
            topic = "ภาพทางการแพทย์"
            target = "Medical Image"
        
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

    # Setup Spine System
    print("Loading Spine ML Model...")
    spine_model = None
    spine_mode = "mock"
    if Path(SPINE_MODEL_PATH).exists():
        try:
            spine_model = keras.models.load_model(SPINE_MODEL_PATH)
            spine_mode = "real"
            print(f"SUCCESS: Spine model loaded: {SPINE_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load spine model: {e}")
    
    if spine_model is None:
        print("INFO: Falling back to Spine mock model...")
        spine_model = MockModel(num_classes=4, is_dental=False)

    try:
        spine_meta = load_metadata(SPINE_METADATA_PATH)
        spine_raw_classes = spine_meta["classes"]
        spine_idx_to_class = {}
        spine_names_map = {
            "infection": "กระดูกสันหลังอักเสบติดเชื้อ (Spinal Infection)",
            "normal": "กระดูกสันหลังปกติ (Normal Spine)",
            "spondyloarthropathy": "กลุ่มโรคข้อกระดูกสันหลังอักเสบ (Spondyloarthropathy)",
            "spondylolisthesis": "กระดูกสันหลังเคลื่อน (Spondylolisthesis)"
        }
        for k, v in spine_raw_classes.items():
            spine_idx_to_class[str(k)] = {
                "code": v.upper(),
                "name": spine_names_map.get(v, v)
            }
    except Exception as e:
        print(f"ERROR: Failed to load spine metadata: {e}")
        spine_idx_to_class = {
            "0": {"code": "INFECTION", "name": "กระดูกสันหลังอักเสบติดเชื้อ (Spinal Infection)"},
            "1": {"code": "NORMAL", "name": "กระดูกสันหลังปกติ (Normal Spine)"},
            "2": {"code": "SPONDYLOARTHROPATHY", "name": "กลุ่มโรคข้อกระดูกสันหลังอักเสบ (Spondyloarthropathy)"},
            "3": {"code": "SPONDYLOLISTHESIS", "name": "กระดูกสันหลังเคลื่อน (Spondylolisthesis)"}
        }

    state["spine"] = {
        "model": spine_model,
        "target_size": (spine_model.input_shape[2], spine_model.input_shape[1]),
        "idx_to_class": spine_idx_to_class,
        "model_mode": spine_mode
    }

    # Setup Osteoporosis System
    print("Loading Osteoporosis ML Model...")
    osteo_model = None
    osteo_mode = "mock"
    if Path(OSTEO_MODEL_PATH).exists():
        try:
            osteo_model = keras.models.load_model(OSTEO_MODEL_PATH)
            osteo_mode = "real"
            print(f"SUCCESS: Osteoporosis model loaded: {OSTEO_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load osteoporosis model: {e}")
    
    if osteo_model is None:
        print("INFO: Falling back to Osteoporosis mock model...")
        osteo_model = MockModel(num_classes=3, is_dental=False)

    try:
        osteo_meta = load_metadata(OSTEO_METADATA_PATH)
        osteo_raw_classes = osteo_meta["classes"]
        osteo_idx_to_class = {}
        osteo_names_map = {
            "Normal": "มวลกระดูกปกติ (Normal)",
            "Osteopenia": "กระดูกบาง (Osteopenia)",
            "Osteoporosis": "กระดูกพรุน (Osteoporosis)"
        }
        for k, v in osteo_raw_classes.items():
            osteo_idx_to_class[str(k)] = {
                "code": v.upper(),
                "name": osteo_names_map.get(v, v)
            }
    except Exception as e:
        print(f"ERROR: Failed to load osteoporosis metadata: {e}")
        osteo_idx_to_class = {
            "0": {"code": "NORMAL", "name": "มวลกระดูกปกติ (Normal)"},
            "1": {"code": "OSTEOPENIA", "name": "กระดูกบาง (Osteopenia)"},
            "2": {"code": "OSTEOPOROSIS", "name": "กระดูกพรุน (Osteoporosis)"}
        }

    state["osteoporosis"] = {
        "model": osteo_model,
        "target_size": (osteo_model.input_shape[2], osteo_model.input_shape[1]),
        "idx_to_class": osteo_idx_to_class,
        "model_mode": osteo_mode
    }

    # Setup Stroke System
    print("Loading Stroke ML Model...")
    stroke_model = None
    stroke_mode = "mock"
    if Path(STROKE_MODEL_PATH).exists():
        try:
            stroke_model = keras.models.load_model(STROKE_MODEL_PATH)
            stroke_mode = "real"
            print(f"SUCCESS: Stroke model loaded: {STROKE_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load stroke model: {e}")
    
    if stroke_model is None:
        print("INFO: Falling back to Stroke mock model...")
        stroke_model = MockModel(num_classes=2, is_dental=False)

    try:
        stroke_meta = load_metadata(STROKE_METADATA_PATH)
        stroke_raw_classes = stroke_meta["classes"]
        stroke_idx_to_class = {}
        stroke_names_map = {
            "Stroke": "สมองขาดเลือด (Stroke)",
            "Normal": "ปกติ (Normal)"
        }
        for k, v in stroke_raw_classes.items():
            stroke_idx_to_class[str(k)] = {
                "code": v.upper(),
                "name": stroke_names_map.get(v, v)
            }
    except Exception as e:
        print(f"ERROR: Failed to load stroke metadata: {e}")
        stroke_idx_to_class = {
            "0": {"code": "STROKE", "name": "สมองขาดเลือด (Stroke)"},
            "1": {"code": "NORMAL", "name": "ปกติ (Normal)"}
        }

    state["stroke"] = {
        "model": stroke_model,
        "target_size": (stroke_model.input_shape[2], stroke_model.input_shape[1]),
        "idx_to_class": stroke_idx_to_class,
        "model_mode": stroke_mode
    }

    print("\n" + "="*60)
    print("UNIFIED MEDICAL AI SERVER STARTED")
    print("="*60)
    print("Unified Portal:  http://localhost:9000/")
    print("Dental System:   http://localhost:9000/dental/")
    print("Skin System:     http://localhost:9000/skin/")
    print("Spine System:    http://localhost:9000/spine/")
    print("Osteoporosis:    http://localhost:9000/osteoporosis/")
    print("Stroke System:   http://localhost:9000/stroke/")
    print("="*60 + "\n")
    yield
    state["dental"].clear()
    state["skin"].clear()
    state["spine"].clear()
    state["osteoporosis"].clear()
    state["stroke"].clear()


app = FastAPI(
    title="Unified Medical AI Portal Server",
    description="Unified API server hosting Dental, Skin, Spine, Osteoporosis, and Stroke diagnostic systems.",
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
        content = content.replace("http://localhost:8002", "/spine/")
        content = content.replace("http://localhost:8003", "/osteoporosis/")
        content = content.replace("http://localhost:8004", "/stroke/")
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


# ---- Spine Frontend ----
@app.get("/spine/", response_class=HTMLResponse)
async def get_spine_page():
    spine_html_path = Path("spine-diagnosis-app/static/index.html")
    if spine_html_path.exists():
        with open(spine_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Spine System</h1><p>Static index.html not found.</p>", status_code=404)


# ---- Osteoporosis Frontend ----
@app.get("/osteoporosis/", response_class=HTMLResponse)
async def get_osteoporosis_page():
    osteo_html_path = Path("osteoporosis-diagnosis-app/static/index.html")
    if osteo_html_path.exists():
        with open(osteo_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Osteoporosis System</h1><p>Static index.html not found.</p>", status_code=404)


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
    
    # Handle single sigmoid output for binary classification (e.g. Stroke model)
    if len(probs) == 1:
        p1 = float(probs[0])  # Normal
        p0 = 1.0 - p1         # Stroke
        probs = np.array([p0, p1], dtype=np.float32)

    # Sort normally first
    predictions_list = [
        {
            "rank": 0,
            "index": int(i),
            "code": idx_to_class.get(str(i), {}).get("code", str(i)),
            "name": idx_to_class.get(str(i), {}).get("name", str(i)),
            "probability": round(float(probs[i]), 6),
            "probability_pct": round(float(probs[i]) * 100, 2),
        }
        for i in range(len(probs))
    ]

    # Special threshold sorting for Stroke model
    if sys_key == "stroke":
        # Index 0 is Stroke, Index 1 is Normal
        stroke_prob = float(probs[0])
        if stroke_prob >= 0.3:
            # Sort Stroke first
            predictions = sorted(predictions_list, key=lambda x: x["index"] == 0, reverse=True)
        else:
            # Sort Normal first
            predictions = sorted(predictions_list, key=lambda x: x["index"] == 1, reverse=True)
    else:
        # Standard sort by probability
        predictions = sorted(predictions_list, key=lambda val: val["probability"], reverse=True)

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


def generate_gradcam_base64(image_bytes: bytes, model, target_size: tuple, is_mock: bool = False) -> str:
    try:
        # Load and preprocess original image
        img_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img_orig.resize(target_size, Image.LANCZOS)
        
        # Keep original image array in BGR for OpenCV
        img_np = np.array(img_resized, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        if is_mock or not hasattr(model, "layers"):
            # Generate a mock heatmap centered on the image for mock model
            height, width = target_size[1], target_size[0]
            x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
            d = np.sqrt(x_grid*x_grid + y_grid*y_grid)
            sigma, mu = 0.5, 0.0
            heatmap_resized = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
        else:
            # Float array for prediction
            x = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
            
            backbone = model.get_layer("efficientnetb0")
            last_conv_layer_name = "top_activation"
            
            # Backbone grad model
            backbone_grad_model = tf.keras.Model(
                inputs=backbone.inputs,
                outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output]
            )
            
            x_in = x
            if "augmentation" in [l.name for l in model.layers]:
                x_in = model.get_layer("augmentation")(x_in)
                
            with tf.GradientTape() as tape:
                conv_outputs, backbone_outputs = backbone_grad_model(x_in)
                tape.watch(conv_outputs)
                
                y = backbone_outputs
                pool_layers = [l for l in model.layers if "pool" in l.name.lower()]
                if pool_layers:
                    y = pool_layers[0](y)
                
                dropout_layers = [l for l in model.layers if "dropout" in l.name.lower()]
                if dropout_layers:
                    y = dropout_layers[0](y)
                    
                dense_layers = [l for l in model.layers if "dense" in l.name.lower()]
                if dense_layers:
                    preds = dense_layers[-1](y)
                else:
                    return ""
                
                # Gradient of Stroke class score (which is 1.0 - preds[0][0]) w.r.t conv features
                stroke_score = 1.0 - preds[0][0]
                
            grads = tape.gradient(stroke_score, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            
            # Weighted sum
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # ReLU and normalization
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
            heatmap_np = heatmap.numpy()
            
            # Resize heatmap
            heatmap_resized = cv2.resize(heatmap_np, (target_size[0], target_size[1]))
        
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Color map
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay heatmap on BGR original
        superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return b64_str
    except Exception as e:
        print(f"ERROR: Failed to generate Grad-CAM: {e}")
        return ""


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


# ==========================================
# 🦴 SPINE APIS (mounted under /spine/*)
# ==========================================

@app.get("/spine/health")
async def spine_health():
    return {"status": "ok", "mode": state["spine"]["model_mode"]}


@app.post("/spine/validate-image")
async def spine_validate_image(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_type = file.content_type.lower()
    is_jpg = (
        content_type in ["image/jpeg", "image/jpg"] or 
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg")
    )
    if not is_jpg:
        raise HTTPException(status_code=400, detail="ระบบรองรับเฉพาะรูปภาพประเภท JPG / JPEG เท่านั้น")
        
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    validation_result = claude_service.validate_image(
        image_bytes, 
        system_type="spine", 
        media_type=file.content_type
    )
    return JSONResponse(validation_result)


@app.post("/spine/predict")
async def spine_predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_type = file.content_type.lower()
    is_jpg = (
        content_type in ["image/jpeg", "image/jpg"] or 
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg")
    )
    if not is_jpg:
        raise HTTPException(status_code=400, detail="ระบบรองรับเฉพาะรูปภาพประเภท JPG / JPEG เท่านั้น")
        
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    res = handle_prediction(image_bytes, "spine")
    res["filename"] = file.filename
    return JSONResponse(res)


@app.post("/spine/treatment-advice")
async def spine_treatment_advice(request: TreatmentRequest):
    try:
        rec = claude_service.generate_treatment_recommendation(
            disease_name=request.disease_name,
            disease_code=request.disease_code,
            confidence=request.confidence,
            patient_info=request.patient_info,
            system_type="spine"
        )
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


@app.post("/spine/chat")
async def spine_chat(request: ChatRequest):
    try:
        resp = claude_service.chat_with_doctor(
            question=request.question,
            conversation_history=request.conversation_history,
            context=request.context,
            system_type="spine"
        )
        return ApiResponse(
            success=True,
            data={"response": resp, "question": request.question},
            message="ตอบคำถามสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/spine/claude-status")
def spine_claude_status():
    return claude_service.test_connection()


# ==========================================
# 🦴 OSTEOPOROSIS APIS (mounted under /osteoporosis/*)
# ==========================================

@app.get("/osteoporosis/health")
async def osteoporosis_health():
    return {"status": "ok", "mode": state["osteoporosis"]["model_mode"]}


@app.post("/osteoporosis/validate-image")
async def osteoporosis_validate_image(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_type = file.content_type.lower()
    is_jpg = (
        content_type in ["image/jpeg", "image/jpg"] or 
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg")
    )
    if not is_jpg:
        raise HTTPException(status_code=400, detail="ระบบรองรับเฉพาะรูปภาพประเภท JPG / JPEG เท่านั้น")
        
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    validation_result = claude_service.validate_image(
        image_bytes, 
        system_type="osteoporosis", 
        media_type=file.content_type
    )
    return JSONResponse(validation_result)


@app.post("/osteoporosis/predict")
async def osteoporosis_predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_type = file.content_type.lower()
    is_jpg = (
        content_type in ["image/jpeg", "image/jpg"] or 
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg")
    )
    if not is_jpg:
        raise HTTPException(status_code=400, detail="ระบบรองรับเฉพาะรูปภาพประเภท JPG / JPEG เท่านั้น")
        
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    res = handle_prediction(image_bytes, "osteoporosis")
    res["filename"] = file.filename
    return JSONResponse(res)


@app.post("/osteoporosis/treatment-advice")
async def osteoporosis_treatment_advice(request: TreatmentRequest):
    try:
        rec = claude_service.generate_treatment_recommendation(
            disease_name=request.disease_name,
            disease_code=request.disease_code,
            confidence=request.confidence,
            patient_info=request.patient_info,
            system_type="osteoporosis"
        )
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


@app.post("/osteoporosis/chat")
async def osteoporosis_chat(request: ChatRequest):
    try:
        resp = claude_service.chat_with_doctor(
            question=request.question,
            conversation_history=request.conversation_history,
            context=request.context,
            system_type="osteoporosis"
        )
        return ApiResponse(
            success=True,
            data={"response": resp, "question": request.question},
            message="ตอบคำถามสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/osteoporosis/claude-status")
def osteoporosis_claude_status():
    return claude_service.test_connection()


# ---- Stroke Frontend ----
@app.get("/stroke/", response_class=HTMLResponse)
async def get_stroke_page():
    stroke_html_path = Path("stroke-diagnosis-app/static/index.html")
    if stroke_html_path.exists():
        with open(stroke_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Stroke System</h1><p>Static index.html not found.</p>", status_code=404)


# ==========================================
# 🧠 STROKE APIS (mounted under /stroke/*)
# ==========================================

@app.get("/stroke/health")
async def stroke_health():
    return {"status": "ok", "mode": state["stroke"]["model_mode"]}


@app.post("/stroke/validate-image")
async def stroke_validate_image(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_type = file.content_type.lower()
    is_jpg = (
        content_type in ["image/jpeg", "image/jpg"] or 
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg")
    )
    if not is_jpg:
        raise HTTPException(status_code=400, detail="ระบบรองรับเฉพาะรูปภาพประเภท JPG / JPEG เท่านั้น")
        
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    validation_result = claude_service.validate_image(
        image_bytes, 
        system_type="stroke", 
        media_type=file.content_type
    )
    return JSONResponse(validation_result)


@app.post("/stroke/predict")
async def stroke_predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_type = file.content_type.lower()
    is_jpg = (
        content_type in ["image/jpeg", "image/jpg"] or 
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg")
    )
    if not is_jpg:
        raise HTTPException(status_code=400, detail="ระบบรองรับเฉพาะรูปภาพประเภท JPG / JPEG เท่านั้น")
        
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    res = handle_prediction(image_bytes, "stroke")
    res["filename"] = file.filename
    
    # Generate Grad-CAM image
    model = state["stroke"]["model"]
    target_size = state["stroke"]["target_size"]
    is_mock = (state["stroke"]["model_mode"] == "mock")
    
    gradcam_b64 = generate_gradcam_base64(image_bytes, model, target_size, is_mock=is_mock)
    res["gradcam_image"] = gradcam_b64
    
    return JSONResponse(res)


@app.post("/stroke/treatment-advice")
async def stroke_treatment_advice(request: TreatmentRequest):
    try:
        rec = claude_service.generate_treatment_recommendation(
            disease_name=request.disease_name,
            disease_code=request.disease_code,
            confidence=request.confidence,
            patient_info=request.patient_info,
            system_type="stroke"
        )
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


@app.post("/stroke/chat")
async def stroke_chat(request: ChatRequest):
    try:
        resp = claude_service.chat_with_doctor(
            question=request.question,
            conversation_history=request.conversation_history,
            context=request.context,
            system_type="stroke"
        )
        return ApiResponse(
            success=True,
            data={"response": resp, "question": request.question},
            message="ตอบคำถามสำเร็จ"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stroke/claude-status")
def stroke_claude_status():
    return claude_service.test_connection()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
