import os
import base64
import requests
import json
import logging
from PIL import Image
import io
from dotenv import load_dotenv
from .advanced_property_validator import AdvancedPropertyValidator

load_dotenv()

logger = logging.getLogger(__name__)

class AdvancedAIModerator:
    def __init__(self):
        self.api_key = os.getenv('XAI_API_KEY')
        self.base_url = "https://api.x.ai/v1"
        self.property_validator = AdvancedPropertyValidator()
        
    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for API upload"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return img_base64
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None

    def analyze_with_grok_vision(self, image_path, text_content=""):
        """
        Use Grok's vision model with strict property and vehicle validation
        """
        try:
            # First, validate if this is a property or vehicle image
            is_property_or_vehicle, property_score, property_reasons = self.property_validator.validate_property_image(image_path)
            
            if not is_property_or_vehicle:
                return self.get_property_rejection_result(property_score, property_reasons)

            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return self.get_fallback_result("Failed to process image")

            prompt = f"""
            CRITICAL: This image has been pre-validated as a property or vehicle image (Score: {property_score}/100).
            You MUST be EXTREMELY STRICT about content moderation for property and vehicle listings.
            Vehicle images (cars, trucks, motorcycles, etc.) are VALID content alongside property images.

            PROPERTY/VEHICLE VALIDATION RESULTS:
            - Validation Score: {property_score}/100
            - Validation Reasons: {', '.join(property_reasons)}

            TEXT CONTEXT: {text_content if text_content else "No text description provided"}

            STRICT CATEGORIES ANALYSIS:

            1. 🔞 EXPLICIT NUDITY (0-100) - ZERO TOLERANCE
               - ANY nudity, genitalia, explicit sexual acts = IMMEDIATE BLOCK (Score 100)
               - Pornographic content = IMMEDIATE BLOCK
               - Fully exposed bodies = IMMEDIATE BLOCK

            2. 🧍 SUGGESTIVE/LINGERIE (0-100) - VERY STRICT
               - People in underwear/bikinis = BLOCK (Score 80+)
               - Sexually suggestive poses = BLOCK (Score 70+)
               - Provocative clothing = BLOCK (Score 60+)

            3. 🤬 OFFENSIVE TEXT (0-100) - ZERO TOLERANCE
               - ANY profanity, swear words = BLOCK (Score 100)
               - Sexual references = BLOCK (Score 80+)
               - Hate speech = BLOCK (Score 100)

            4. 💉 VIOLENCE/BLOOD (0-100) - ZERO TOLERANCE
               - ANY weapons, violence = BLOCK (Score 100)
               - Blood, injuries = BLOCK (Score 100)

            5. 💀 HATE/RACIAL CONTENT (0-100) - ZERO TOLERANCE
               - ANY racist symbols = BLOCK (Score 100)
               - Discriminatory content = BLOCK (Score 100)

            6. 🏘️ PROPERTY/VEHICLE RELEVANCE (0-100) - ALREADY VALIDATED
               - Score: {property_score}/100
               - Valid content includes: houses, apartments, rooms, vehicles (cars, trucks, motorcycles, etc.)

            ALLOWED CONTENT (BE LIBERAL):
            ✅ Apartments, houses, buildings
            ✅ Interior design, rooms, architecture
            ✅ Empty rooms with furniture
            ✅ Kitchen, bathroom, bedroom (empty)
            ✅ Building exteriors, facades
            ✅ Property landscapes, gardens
            ✅ Professional real estate photos
            ✅ Vehicles (cars, trucks, motorcycles, bicycles, buses, boats)
            ✅ "For Sale", "For Rent" signs
            ✅ Floor plans, blueprints

            RESPONSE FORMAT (STRICT JSON ONLY):
            {{
                "overall_risk_score": 0-100,
                "risk_level": "low/medium/high",
                "block_reason": "None or specific reason",
                "categories": {{
                    "explicit_nudity": {{"score": 0-100, "reason": "detailed explanation"}},
                    "suggestive_content": {{"score": 0-100, "reason": "detailed explanation"}},
                    "offensive_text": {{"score": 0-100, "reason": "detailed explanation"}},
                    "violence": {{"score": 0-100, "reason": "detailed explanation"}},
                    "hate_content": {{"score": 0-100, "reason": "detailed explanation"}},
                    "property_relevance": {{"score": {property_score}, "reason": "Pre-validated: {', '.join(property_reasons)}"}}
                }},
                "moderation_decision": "approve/review/block",
                "confidence": "high/medium/low",
                "detailed_findings": "Comprehensive analysis",
                "property_validation": {{
                    "is_valid_property_or_vehicle": true,
                    "validation_score": {property_score},
                    "validation_reasons": {property_reasons}
                }}
            }}

            BE EXTREMELY VIGILANT: This is for PROPERTY AND VEHICLE LISTINGS. Block ANY inappropriate or NSFW content.
            """

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "model": "grok-2-vision-1212",
                "max_tokens": 1500,
                "temperature": 0.1,
                "stream": False
            }

            logger.info(f"Sending validated property/vehicle image to Grok Vision (Score: {property_score})")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"Grok Vision Response: {content}")
                
                moderation_result = self.extract_json_from_response(content)
                
                if moderation_result:
                    validated_result = self.validate_moderation_result(moderation_result)
                    validated_result["property_validation"] = {
                        "is_valid_property_or_vehicle": True,
                        "validation_score": property_score,
                        "validation_reasons": property_reasons
                    }
                    return validated_result
                else:
                    logger.error("Failed to parse Grok Vision response")
                    return self.get_fallback_with_property(property_score, property_reasons)
            else:
                logger.error(f"Grok Vision API error: {response.status_code}")
                return self.get_fallback_with_property(property_score, property_reasons)
                
        except Exception as e:
            logger.error(f"Grok Vision analysis error: {str(e)}")
            return self.get_fallback_with_property(0, [f"Analysis error: {str(e)}"])

    def get_property_rejection_result(self, property_score, property_reasons):
        """Result for non-property and non-vehicle images"""
        return {
            "overall_risk_score": 100,
            "risk_level": "high",
            "block_reason": "Not a valid property or vehicle image",
            "categories": {
                "explicit_nudity": {"score": 0, "reason": "Image rejected - not property or vehicle related"},
                "suggestive_content": {"score": 0, "reason": "Image rejected - not property or vehicle related"},
                "offensive_text": {"score": 0, "reason": "Image rejected - not property or vehicle related"},
                "violence": {"score": 0, "reason": "Image rejected - not property or vehicle related"},
                "hate_content": {"score": 0, "reason": "Image rejected - not property or vehicle related"},
                "property_relevance": {"score": property_score, "reason": f"Not property/vehicle image: {', '.join(property_reasons)}"}
            },
            "moderation_decision": "block",
            "confidence": "high",
            "detailed_findings": f"This image does not appear to be a property or vehicle listing. Validation score: {property_score}/100. Reasons: {', '.join(property_reasons)}",
            "property_validation": {
                "is_valid_property_or_vehicle": False,
                "validation_score": property_score,
                "validation_reasons": property_reasons
            }
        }

    def extract_json_from_response(self, text):
        """Extract JSON from AI response"""
        try:
            text = text.strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
        return None

    def validate_moderation_result(self, result):
        """Validate and enhance moderation result"""
        try:
            required_categories = [
                "explicit_nudity", "suggestive_content", "offensive_text", 
                "violence", "hate_content", "property_relevance"
            ]
            
            if "categories" not in result:
                result["categories"] = {}
            
            for category in required_categories:
                if category not in result["categories"]:
                    result["categories"][category] = {"score": 0, "reason": "Not analyzed"}
                else:
                    cat_data = result["categories"][category]
                    if "score" not in cat_data:
                        cat_data["score"] = 0
                    else:
                        cat_data["score"] = max(0, min(100, int(cat_data["score"])))
                    
                    if "reason" not in cat_data:
                        cat_data["reason"] = "No specific reason provided"
            
            if "overall_risk_score" not in result:
                risk_scores = [
                    result["categories"]["explicit_nudity"]["score"],
                    result["categories"]["suggestive_content"]["score"],
                    result["categories"]["offensive_text"]["score"],
                    result["categories"]["violence"]["score"],
                    result["categories"]["hate_content"]["score"]
                ]
                result["overall_risk_score"] = max(risk_scores)
            
            risk_score = result["overall_risk_score"]
            if risk_score >= 70:
                result["risk_level"] = "high"
                result["moderation_decision"] = "block"
            elif risk_score >= 30:
                result["risk_level"] = "medium"
                result["moderation_decision"] = "review"
            else:
                result["risk_level"] = "low"
                result["moderation_decision"] = "approve"
            
            if result["moderation_decision"] != "approve":
                high_risk_cats = []
                for cat_name, cat_data in result["categories"].items():
                    if cat_name != "property_relevance" and cat_data["score"] >= 50:
                        high_risk_cats.append(cat_name.replace('_', ' ').title())
                
                if high_risk_cats:
                    result["block_reason"] = f"High risk in: {', '.join(high_risk_cats)}"
                else:
                    result["block_reason"] = "Content requires review"
            else:
                result["block_reason"] = "None"
            
            if "confidence" not in result:
                result["confidence"] = "high" if risk_score < 30 else "medium"
            
            return result
            
        except Exception as e:
            logger.error(f"Result validation error: {e}")
            return self.get_fallback_result("Result validation failed")

    def get_fallback_result(self, reason):
        """Basic fallback result"""
        return {
            "overall_risk_score": 50,
            "risk_level": "medium",
            "block_reason": reason,
            "categories": {
                "explicit_nudity": {"score": 0, "reason": "Analysis failed"},
                "suggestive_content": {"score": 0, "reason": "Analysis failed"},
                "offensive_text": {"score": 0, "reason": "Analysis failed"},
                "violence": {"score": 0, "reason": "Analysis failed"},
                "hate_content": {"score": 0, "reason": "Analysis failed"},
                "property_relevance": {"score": 50, "reason": "Analysis failed"}
            },
            "moderation_decision": "review",
            "confidence": "low",
            "detailed_findings": f"AI analysis failed: {reason}. Manual review required.",
            "property_validation": {
                "is_valid_property_or_vehicle": False,
                "validation_score": 0,
                "validation_reasons": ["Analysis system error"]
            }
        }

    def get_fallback_with_property(self, property_score, property_reasons):
        """Fallback with property/vehicle validation data"""
        result = self.get_fallback_result("AI service unavailable")
        result["property_validation"] = {
            "is_valid_property_or_vehicle": property_score >= 70,
            "validation_score": property_score,
            "validation_reasons": property_reasons
        }
        
        if not result["property_validation"]["is_valid_property_or_vehicle"]:
            result["moderation_decision"] = "block"
            result["overall_risk_score"] = 100
            result["risk_level"] = "high"
            result["block_reason"] = "Not a valid property or vehicle image"
        
        return result