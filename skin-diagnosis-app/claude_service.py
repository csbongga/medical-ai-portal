"""
Claude API Service สำหรับระบบช่วยวินิจฉัยโรคผิวหนัง
ให้คำแนะนำการรักษาและตอบคำถามแพทย์ผิวหนัง
"""
import os
import json
import base64
from typing import List, Dict, Any
from dotenv import load_dotenv
import anthropic

# โหลด environment variables
load_dotenv()

class ClaudeService:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("กรุณาตั้งค่า ANTHROPIC_API_KEY ใน .env file")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6"  # รุ่นล่าสุด

        # System prompt สำหรับการให้คำแนะนำการรักษา
        self.medical_system_prompt = """คุณเป็นระบบ AI ช่วยสนับสนุนการตัดสินใจของแพทย์ผิวหนัง โดยมีหน้าที่ให้คำแนะนำทางการแพทย์อ้างอิงหลักฐานทางวิทยาศาสตร์

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

    def generate_treatment_recommendation(
        self,
        disease_name: str,
        disease_code: str,
        confidence: float,
        patient_info: str = None
    ) -> str:
        """
        สร้างคำแนะนำการรักษาจากผลการทำนาย

        Args:
            disease_name: ชื่อโรคที่ทำนายได้
            disease_code: รหัสโรค (เช่น IP, IT, PD)
            confidence: ระดับความเชื่อมั่นเป็น % (เช่น 89.5)
            patient_info: ข้อมูลผู้ป่วยเพิ่มเติม (ถ้ามี)

        Returns:
            คำแนะนำการรักษาเป็นภาษาไทย
        """

        # สร้าง prompt สำหรับการให้คำแนะนำ
        user_prompt = f"""โปรดวิเคราะห์และให้คำแนะนำการรักษาจากผลการทำนายต่อไปนี้:

📊 ผลการทำนายจากระบบ AI:
- โรคที่ตรวจพบ: {disease_name}
- รหัสโรค: {disease_code}
- ระดับความเชื่อมั่น: {confidence:.1f}%

"""

        if patient_info:
            user_prompt += f"""
👤 ข้อมูลผู้ป่วยเพิ่มเติม:
{patient_info}
"""

        user_prompt += f"""
กรุณาตอบเป็น JSON format เท่านั้น ตามโครงสร้างที่กำหนด:

ตัวอย่าง JSON:
{{
  "causes": "การอักเสบของโพรงประสาทฟันที่ลุกลามถึงเนื้อเยื่อรอบปลายรากฟัน มักเกิดจากฟันผุลึก",
  "treatments": [
    {{"type": "warning", "number": "!", "label": "ประเมินเบื้องต้น", "description": "ทดสอบชีพจรฟันและประเมินระดับความเจ็บปวด"}},
    {{"type": "primary", "number": "1", "label": "การรักษาหลัก", "description": "รักษารากฟัน (Root Canal Treatment) โดยเร่งด่วน"}},
    {{"type": "secondary", "number": "2", "label": "การจัดการอาการ", "description": "ให้ยาแก้ปวดและยาปฏิชีวนะตามความจำเป็น"}},
    {{"type": "secondary", "number": "3", "label": "ติดตามผล", "description": "ตรวจซ้ำและถ่าย X-ray หลังรักษา 3-6 เดือน"}}
  ]
}}

สำคัญมาก:
- ตอบเป็น JSON object เท่านั้น ห้ามเพิ่มข้อความอื่น
- type ใช้ได้เฉพาะ: "warning", "primary", "secondary"
- ระดับความเชื่อมั่น: {confidence:.1f}%"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,  # เพิ่มเพื่อให้ Claude ตอบจบ
                temperature=0.3,  # ค่าต่ำเพื่อความสม่ำเสมอ
                system=self.medical_system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Get response text and clean it
            response_text = response.content[0].text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```

            response_text = response_text.strip()

            try:
                # Try to parse as JSON first
                json_response = json.loads(response_text)
                return json.dumps(json_response, ensure_ascii=False)
            except json.JSONDecodeError:
                # If not valid JSON, return the raw response for fallback handling
                return response_text

        except Exception as e:
            return json.dumps({"error": f"เกิดข้อผิดพลาดในการเชื่อมต่อ Claude API: {str(e)}"}, ensure_ascii=False)

    def chat_with_doctor(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """
        ตอบคำถามของแพทย์ในรูปแบบการสนทนา

        Args:
            question: คำถามของแพทย์
            conversation_history: ประวัติการสนทนาก่อนหน้า
            context: ข้อมูลบริบท (ผลทำนาย, ข้อมูลผู้ป่วย)

        Returns:
            คำตอบเป็นภาษาไทย
        """

        # สร้าง system prompt สำหรับ chat (แยกจาก medical system prompt)
        chat_system = """คุณเป็น AI ผู้ช่วยแพทย์ผิวหนังที่เป็นมิตรและช่วยเหลือ โดยมีหน้าที่ตอบคำถามและให้คำแนะนำเบื้องต้น

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

        # เตรียม messages สำหรับการสนทนา
        messages = []

        # เพิ่มบริบทเริ่มต้น
        if context:
            context_msg = "📋 ข้อมูลบริบทปัจจุบัน:\n"
            if context.get('disease'):
                context_msg += f"- โรคที่ตรวจพบ: {context['disease']}\n"
            if context.get('confidence'):
                context_msg += f"- ความเชื่อมั่น: {context['confidence']}%\n"
            if context.get('patient_info'):
                context_msg += f"- ข้อมูลผู้ป่วย: {context['patient_info']}\n"

            messages.append({"role": "user", "content": context_msg})
            messages.append({"role": "assistant", "content": "รับทราบข้อมูลบริบทแล้วครับ มีอะไรให้ช่วยเหลือไหมครับ?"})

        # เพิ่มประวัติการสนทนา
        if conversation_history:
            messages.extend(conversation_history)

        # เพิ่มคำถามปัจจุบัน
        messages.append({"role": "user", "content": question})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.4,
                system=chat_system,
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ Claude API: {str(e)}"

    def validate_skin_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        ตรวจสอบว่าภาพที่อัปโหลดเป็นภาพผิวหนังหรือไม่

        Args:
            image_bytes: ข้อมูลภาพในรูปแบบ bytes

        Returns:
            Dict ที่มี is_valid และ message
        """
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Create message with image
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": """โปรดวิเคราะห์ภาพนี้และตอบเป็น JSON format เท่านั้น:

ตรวจสอบว่าภาพนี้เป็น:
1. ภาพผิวหนัง (Skin lesion image) หรือไม่
2. มีรอยโรคผิวหนัง ไฝ หรือแผลบนผิวหนังปรากฏหรือไม่
3. เป็นภาพทางการแพทย์ที่เกี่ยวข้องกับผิวหนังวิทยาหรือไม่

ตอบเป็น JSON format เท่านั้น:
{
  "is_valid": true/false,
  "confidence": 0-100,
  "message": "คำอธิบายภาษาไทย"
}

หากเป็นภาพผิวหนัง: is_valid = true
หากไม่ใช่ภาพผิวหนัง: is_valid = false"""
                    }
                ]
            }

            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,  # ต่ำมากเพื่อความแม่นยำ
                messages=[message]
            )

            response_text = response.content[0].text.strip()

            # Clean response
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
                    "message": "ไม่สามารถวิเคราะห์ภาพได้ กรุณาอัปโหลดภาพผิวหนังที่ชัดเจน"
                }

        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0,
                "message": f"เกิดข้อผิดพลาดในการตรวจสอบภาพ: {str(e)}"
            }

    def test_connection(self) -> Dict[str, Any]:
        """ทดสอบการเชื่อมต่อ Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": "สวัสดี"}]
            )
            return {
                "status": "success",
                "message": "เชื่อมต่อ Claude API สำเร็จ",
                "model": self.model,
                "response": response.content[0].text
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"เชื่อมต่อ Claude API ไม่สำเร็จ: {str(e)}"
            }

# สร้าง instance global
claude_service = ClaudeService()