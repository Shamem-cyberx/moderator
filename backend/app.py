from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from utils.image_utils import (
    allowed_file, 
    moderate_content_advanced,
    save_uploaded_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'blocked'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'valid_uploads'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'manual_review'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'rejected_non_property'), exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        text_content = request.form.get('text_content', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], 'valid_uploads', filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
            
            # Save to temporary location first
            file.save(temp_path)
            
            try:
                # Advanced AI content moderation with strict property/vehicle validation
                logger.info("Starting advanced AI moderation with strict property/vehicle validation...")
                moderation_result = moderate_content_advanced(temp_path, text_content)
                
                # Check property/vehicle validation first
                property_validation = moderation_result.get('property_validation', {})
                is_valid_property_or_vehicle = property_validation.get('is_valid_property_or_vehicle', False)
                property_score = property_validation.get('validation_score', 0)
                
                # Process moderation decision
                decision = moderation_result.get('moderation_decision', 'review')
                risk_score = moderation_result.get('overall_risk_score', 0)
                
                logger.info(f"Moderation result - Property/Vehicle Valid: {is_valid_property_or_vehicle}, Score: {property_score}, Decision: {decision}, Risk: {risk_score}")
                
                # STRICT PROPERTY/VEHICLE VALIDATION - Reject non-property/vehicle images immediately
                if not is_valid_property_or_vehicle:
                    rejected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'rejected_non_property', filename)
                    os.rename(temp_path, rejected_path)
                    
                    return jsonify({
                        'success': False,
                        'error': 'Not a property or vehicle image',
                        'message': f'Image rejected. Property/Vehicle validation score: {property_score}/100',
                        'decision': 'rejected_non_property',
                        'property_score': property_score,
                        'property_reasons': property_validation.get('validation_reasons', []),
                        'risk_score': risk_score,
                        'risk_level': 'high',
                        'categories': moderation_result.get('categories', {}),
                        'detailed_explanation': moderation_result.get('detailed_findings', ''),
                        'analysis_method': 'strict_property_vehicle_validation'
                    }), 400
                
                if decision == "block" or risk_score >= 70:
                    # Move to blocked folder
                    blocked_path = os.path.join(app.config['UPLOAD_FOLDER'], 'blocked', filename)
                    os.rename(temp_path, blocked_path)
                    
                    return jsonify({
                        'success': False,
                        'error': 'Content violation detected',
                        'message': moderation_result.get('block_reason', 'Content violates guidelines'),
                        'decision': 'blocked',
                        'property_score': property_score,
                        'risk_score': risk_score,
                        'risk_level': moderation_result.get('risk_level', 'high'),
                        'categories': moderation_result.get('categories', {}),
                        'detailed_explanation': moderation_result.get('detailed_findings', ''),
                        'confidence': moderation_result.get('confidence', 'medium'),
                        'property_reasons': property_validation.get('validation_reasons', []),
                        'analysis_method': 'grok_vision_advanced'
                    }), 400
                
                elif decision == "review" or risk_score >= 30:
                    # Move to manual review
                    review_path = os.path.join(app.config['UPLOAD_FOLDER'], 'manual_review', filename)
                    os.rename(temp_path, review_path)
                    
                    return jsonify({
                        'success': False,
                        'error': 'Content requires manual review',
                        'message': 'Content flagged for review by moderation team',
                        'decision': 'manual_review',
                        'property_score': property_score,
                        'risk_score': risk_score,
                        'risk_level': moderation_result.get('risk_level', 'medium'),
                        'categories': moderation_result.get('categories', {}),
                        'detailed_explanation': moderation_result.get('detailed_findings', ''),
                        'confidence': moderation_result.get('confidence', 'medium'),
                        'property_reasons': property_validation.get('validation_reasons', []),
                        'analysis_method': 'grok_vision_advanced'
                    }), 400
                
                else:
                    # Content is safe and valid property/vehicle - move to valid uploads
                    os.rename(temp_path, final_path)
                    
                    return jsonify({
                        'success': True,
                        'message': 'Property or vehicle image uploaded successfully',
                        'filename': filename,
                        'decision': 'approved',
                        'property_score': property_score,
                        'risk_score': risk_score,
                        'risk_level': 'low',
                        'categories': moderation_result.get('categories', {}),
                        'moderation_notes': 'Content approved by advanced AI moderation',
                        'property_reasons': property_validation.get('validation_reasons', []),
                        'analysis_method': 'grok_vision_advanced'
                    }), 200
            
            except Exception as analysis_error:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                logger.error(f"Image analysis error: {str(analysis_error)}")
                return jsonify({
                    'success': False,
                    'error': 'Analysis failed',
                    'message': str(analysis_error),
                    'decision': 'error'
                }), 500
        
        return jsonify({
            'success': False,
            'error': 'Invalid file type',
            'message': 'Only PNG, JPG, JPEG, GIF files are allowed',
            'decision': 'rejected'
        }), 400
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e),
            'decision': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'Strict Property and Vehicle Image Moderation API',
        'version': '5.0.0',
        'features': [
            'Strict Property and Vehicle Validation',
            'Advanced Computer Vision', 
            'Grok Vision AI Analysis',
            'Mathematical Feature Analysis'
        ]
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get upload statistics"""
    try:
        uploads_dir = app.config['UPLOAD_FOLDER']
        valid_dir = os.path.join(uploads_dir, 'valid_uploads')
        blocked_dir = os.path.join(uploads_dir, 'blocked')
        review_dir = os.path.join(uploads_dir, 'manual_review')
        rejected_dir = os.path.join(uploads_dir, 'rejected_non_property')
        
        approved_count = len([f for f in os.listdir(valid_dir) 
                            if os.path.isfile(os.path.join(valid_dir, f))]) if os.path.exists(valid_dir) else 0
        
        blocked_count = len([f for f in os.listdir(blocked_dir) 
                           if os.path.isfile(os.path.join(blocked_dir, f))]) if os.path.exists(blocked_dir) else 0
        
        review_count = len([f for f in os.listdir(review_dir) 
                          if os.path.isfile(os.path.join(review_dir, f))]) if os.path.exists(review_dir) else 0
        
        rejected_count = len([f for f in os.listdir(rejected_dir) 
                            if os.path.isfile(os.path.join(rejected_dir, f))]) if os.path.exists(rejected_dir) else 0
        
        total_processed = approved_count + blocked_count + review_count + rejected_count
        
        return jsonify({
            'approved_property': approved_count,
            'blocked_violations': blocked_count,
            'manual_review': review_count,
            'rejected_non_property': rejected_count,
            'total_processed': total_processed,
            'property_approval_rate': f"{(approved_count / max(approved_count + rejected_count, 1)) * 100:.1f}%" if (approved_count + rejected_count) > 0 else "0%",
            'total_approval_rate': f"{(approved_count / max(total_processed, 1)) * 100:.1f}%" if total_processed > 0 else "0%"
        }), 200
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': 'Could not retrieve stats'}), 500

if __name__ == '__main__':
    logger.info("Starting Strict Property and Vehicle Image Moderation API...")
    logger.info("Features: Strict Property/Vehicle Validation + Advanced CV + Grok Vision")
    app.run(debug=True, port=5000, host='0.0.0.0')