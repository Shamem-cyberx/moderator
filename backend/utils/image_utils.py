import os
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, filename, upload_folder):
    """Save uploaded file with temporary naming"""
    temp_filename = f"temp_{filename}"
    temp_path = os.path.join(upload_folder, temp_filename)
    final_path = os.path.join(upload_folder, 'valid_uploads', filename)
    file.save(temp_path)
    return final_path, temp_path

def moderate_content_advanced(image_path, text_content=""):
    """
    Advanced content moderation with strict property and vehicle validation
    """
    try:
        from utils.ai_moderator import AdvancedAIModerator
        moderator = AdvancedAIModerator()
        analysis_result = moderator.analyze_with_grok_vision(image_path, text_content)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Advanced moderation error: {str(e)}")
        return get_fallback_analysis()

def get_fallback_analysis():
    """Fallback analysis"""
    return {
        "overall_risk_score": 50,
        "risk_level": "medium",
        "block_reason": "Moderation system error",
        "categories": {
            "explicit_nudity": {"score": 0, "reason": "System error"},
            "suggestive_content": {"score": 0, "reason": "System error"},
            "offensive_text": {"score": 0, "reason": "System error"},
            "violence": {"score": 0, "reason": "System error"},
            "hate_content": {"score": 0, "reason": "System error"},
            "property_relevance": {"score": 0, "reason": "System error"}
        },
        "moderation_decision": "review",
        "confidence": "low",
        "detailed_findings": "Manual review required due to system error",
        "property_validation": {
            "is_valid_property_or_vehicle": False,
            "validation_score": 0,
            "validation_reasons": ["System error"]
        }
    }