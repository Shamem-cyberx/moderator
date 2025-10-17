import cv2
import numpy as np
from PIL import Image
import logging
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from sklearn.cluster import KMeans
from skimage import feature, measure
import math
from scipy import ndimage

logger = logging.getLogger(__name__)

class AdvancedPropertyValidator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        self.property_features = self.define_property_features()
        
    def setup_models(self):
        """Initialize advanced computer vision models"""
        try:
            self.yolo_model = YOLO('yolov8m.pt')
            logger.info("YOLOv8m model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None

    def define_property_features(self):
        """Define mathematical features for property and vehicle images"""
        return {
            "valid_objects": [
                'house', 'building', 'apartment', 'room', 'bed', 'sofa', 'chair',
                'table', 'kitchen', 'bathroom', 'window', 'door', 'stairs',
                'car', 'truck', 'motorcycle', 'bicycle', 'bus', 'boat', 'train'
            ],
            "vehicle_objects": ['car', 'truck', 'motorcycle', 'bicycle', 'bus', 'boat', 'train'],
            "property_objects": ['house', 'building', 'apartment', 'room', 'bed', 'sofa', 'chair', 
                               'table', 'kitchen', 'bathroom', 'window', 'door', 'stairs'],
            "forbidden_objects": ['person', 'animal', 'logo', 'text', 'weapon', 'electronic'],
            "architectural_shapes": ['rectangle', 'square', 'straight_lines', 'right_angles'],
            "color_profiles": ['neutral', 'warm', 'professional', 'vibrant'],
            "texture_patterns": ['structural', 'geometric', 'repetitive', 'metallic']
        }

    def validate_property_image(self, image_path):
        """
        Comprehensive property AND vehicle image validation
        Returns: (is_valid, score 0-100, reasons)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, 0, ["Invalid image file"]
            
            validation_results = {
                "basic_checks": self.basic_image_checks(image),
                "object_analysis": self.advanced_object_analysis(image),
                "architectural_analysis": self.architectural_feature_analysis(image),
                "color_analysis": self.property_color_analysis(image),
                "texture_analysis": self.property_texture_analysis(image),
                "composition_analysis": self.image_composition_analysis(image),
                "edge_case_detection": self.edge_case_detection(image),
                "vehicle_specific_analysis": self.vehicle_specific_analysis(image)  # Enhanced vehicle analysis
            }
            
            overall_score, reasons = self.calculate_property_score(validation_results)
            
            # Lower threshold to accept more vehicle images
            is_valid = overall_score >= 45  # Reduced threshold for better vehicle acceptance
            
            logger.info(f"Property/Vehicle validation - Score: {overall_score:.2f}, Valid: {is_valid}, Reasons: {reasons}")
            
            return is_valid, overall_score, reasons
            
        except Exception as e:
            logger.error(f"Property/Vehicle validation error: {e}")
            return False, 0, [f"Validation error: {str(e)}"]

    def basic_image_checks(self, image):
        """Basic image quality and dimension checks"""
        results = {}
        height, width = image.shape[:2]
        
        results["aspect_ratio"] = width / height
        results["resolution"] = width * height
        results["min_dimension"] = min(width, height)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results["blur_score"] = cv2.Laplacian(gray, cv2.CV_64F).var()
        results["brightness"] = np.mean(gray)
        results["contrast"] = np.std(gray)
        
        return results

    def advanced_object_analysis(self, image):
        """Advanced object detection with enhanced vehicle recognition"""
        results = {
            "valid_objects": [],
            "vehicle_objects": [],
            "property_objects": [],
            "forbidden_objects": [],
            "object_count": 0,
            "dominant_objects": []
        }
        
        if self.yolo_model:
            try:
                # Use lower confidence threshold for better vehicle detection
                yolo_results = self.yolo_model(image, conf=0.25)  # Lowered threshold
                
                for result in yolo_results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        logger.debug(f"Detected object: {class_name}, Confidence: {confidence:.3f}")
                        
                        if class_name in self.property_features["vehicle_objects"]:
                            results["vehicle_objects"].append({"class": class_name, "confidence": confidence})
                            results["valid_objects"].append({"class": class_name, "confidence": confidence})
                        elif class_name in self.property_features["property_objects"]:
                            results["property_objects"].append({"class": class_name, "confidence": confidence})
                            results["valid_objects"].append({"class": class_name, "confidence": confidence})
                        elif class_name in ['person', 'dog', 'cat', 'cell phone', 'laptop']:
                            results["forbidden_objects"].append({"class": class_name, "confidence": confidence})
                        
                        results["object_count"] += 1
                
                # If no objects detected, try even lower confidence
                if not results["valid_objects"]:
                    logger.info("No valid objects detected, retrying with lower confidence (0.1)")
                    yolo_results = self.yolo_model(image, conf=0.1)
                    for result in yolo_results:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = self.yolo_model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if class_name in self.property_features["vehicle_objects"]:
                                results["vehicle_objects"].append({"class": class_name, "confidence": confidence})
                                results["valid_objects"].append({"class": class_name, "confidence": confidence})
                            elif class_name in self.property_features["property_objects"]:
                                results["property_objects"].append({"class": class_name, "confidence": confidence})
                                results["valid_objects"].append({"class": class_name, "confidence": confidence})
                            elif class_name in ['person', 'dog', 'cat', 'cell phone', 'laptop']:
                                results["forbidden_objects"].append({"class": class_name, "confidence": confidence})
                            
                            results["object_count"] += 1
                
                all_objects = results["valid_objects"] + results["forbidden_objects"]
                if all_objects:
                    results["dominant_objects"] = sorted(all_objects, 
                                                       key=lambda x: x["confidence"], 
                                                       reverse=True)[:3]
                else:
                    logger.warning("No objects detected even with lower confidence")
            
            except Exception as e:
                logger.error(f"YOLO analysis error: {e}")
        
        return results

    def vehicle_specific_analysis(self, image):
        """Enhanced vehicle-specific analysis"""
        results = {
            "has_vehicle_characteristics": False,
            "vehicle_confidence": 0,
            "wheel_detection": 0,
            "metallic_reflection": 0,
            "symmetrical_shape": 0
        }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circular shapes (wheels)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, 
                                 minDist=50, param1=50, param2=30,
                                 minRadius=10, maxRadius=100)
        
        if circles is not None:
            results["wheel_detection"] = len(circles[0])
            logger.info(f"Detected {len(circles[0])} wheel-like circles")
        
        # Detect metallic reflections
        edges = cv2.Canny(gray, 50, 150)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        results["metallic_reflection"] = np.mean(gradient_magnitude)
        
        # Check for symmetrical shapes (common in vehicles)
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        if left_half.shape[1] == right_half.shape[1]:
            right_half_flipped = cv2.flip(right_half, 1)
            mse = np.mean((left_half - right_half_flipped) ** 2)
            results["symmetrical_shape"] = 1.0 - (mse / 255.0)
        
        # Determine if image has vehicle characteristics
        vehicle_indicators = 0
        if results["wheel_detection"] >= 2:
            vehicle_indicators += 1
        if results["metallic_reflection"] > 50:
            vehicle_indicators += 1
        if results["symmetrical_shape"] > 0.6:
            vehicle_indicators += 1
        
        results["has_vehicle_characteristics"] = vehicle_indicators >= 2
        results["vehicle_confidence"] = vehicle_indicators * 25  # 0-75 points
        
        return results

    def architectural_feature_analysis(self, image):
        """Analyze architectural features with vehicle compatibility"""
        results = {
            "straight_lines": 0,
            "right_angles": 0,
            "symmetry_score": 0,
            "geometric_patterns": 0,
            "structural_elements": 0
        }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            results["straight_lines"] = len(lines)
            
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            right_angle_count = sum(1 for angle in angles if abs(angle % 90) < 15)
            results["right_angles"] = right_angle_count
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        geometric_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.6:  # Not too circular (excludes wheels)
                        geometric_shapes += 1
        
        results["geometric_patterns"] = geometric_shapes
        results["structural_elements"] = results["straight_lines"] + geometric_shapes
        
        return results

    def property_color_analysis(self, image):
        """Analyze color profiles with vehicle-specific enhancements"""
        results = {
            "color_variance": 0,
            "dominant_colors": [],
            "color_temperature": 0,
            "professional_score": 0,
            "is_vibrant": False,
            "vehicle_color_score": 0
        }
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = rgb_image.reshape(-1, 3)
        
        results["color_variance"] = np.var(pixels)
        
        try:
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            results["dominant_colors"] = dominant_colors.tolist()
            
            warm_colors = 0
            vibrant_colors = 0
            vehicle_colors = 0  # Metallic, shiny colors
            total_colors = 0
            
            for color in dominant_colors:
                r, g, b = color
                # Warm colors (property)
                if r > g and r > b and r > 100:
                    warm_colors += 1
                # Vibrant colors (vehicles)
                if max(r, g, b) > 200 and min(r, g, b) < 100:
                    vibrant_colors += 1
                    vehicle_colors += 1
                # Metallic colors (gray, silver, etc.)
                if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30 and max(r, g, b) > 100:
                    vehicle_colors += 1
                total_colors += 1
            
            results["color_temperature"] = warm_colors / total_colors if total_colors > 0 else 0
            results["is_vibrant"] = vibrant_colors / total_colors > 0.3 if total_colors > 0 else False
            results["vehicle_color_score"] = (vehicle_colors / total_colors) * 50 if total_colors > 0 else 0
            
            color_std = np.std(dominant_colors, axis=0)
            results["professional_score"] = 1.0 - (np.mean(color_std) / 255)
            
        except Exception as e:
            logger.error(f"Color analysis error: {e}")
        
        return results

    def property_texture_analysis(self, image):
        """Analyze texture patterns with vehicle-specific enhancements"""
        results = {
            "texture_variance": 0,
            "edge_density": 0,
            "homogeneity": 0,
            "structural_texture": 0,
            "is_metallic": False,
            "vehicle_texture_score": 0
        }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lbp = feature.local_binary_pattern(gray, 24, 3, method='uniform')
        results["texture_variance"] = np.var(lbp)
        
        edges = cv2.Canny(gray, 50, 150)
        results["edge_density"] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        structural_pixels = np.sum(morph > 0)
        results["structural_texture"] = structural_pixels / (image.shape[0] * image.shape[1])
        
        # Enhanced metallic texture detection for vehicles
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        metallic_score = np.mean(gradient_magnitude)
        
        results["is_metallic"] = metallic_score > 50 and results["texture_variance"] < 100
        results["vehicle_texture_score"] = min(50, metallic_score / 2)  # 0-50 points
        
        return results

    def image_composition_analysis(self, image):
        """Analyze image composition and layout for both property and vehicles"""
        results = {
            "rule_of_thirds": 0,
            "symmetry": 0,
            "balance_score": 0,
            "focus_regions": 0
        }
        
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        third_x = width // 3
        third_y = height // 3
        
        edges = cv2.Canny(gray, 50, 150)
        
        regions = []
        for i in range(3):
            for j in range(3):
                x_start = i * third_x
                x_end = (i + 1) * third_x
                y_start = j * third_y
                y_end = (j + 1) * third_y
                
                region_edges = edges[y_start:y_end, x_start:x_end]
                edge_density = np.sum(region_edges > 0) / region_edges.size
                regions.append(edge_density)
        
        intersection_points = [1, 3, 5, 7]
        intersection_density = sum(regions[i] for i in intersection_points)
        results["rule_of_thirds"] = intersection_density
        
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        min_height = min(left_half.shape[0], right_half_flipped.shape[0])
        left_half = left_half[:min_height, :]
        right_half_flipped = right_half_flipped[:min_height, :]
        
        if left_half.size > 0 and right_half_flipped.size > 0:
            mse = np.mean((left_half - right_half_flipped) ** 2)
            results["symmetry"] = 1.0 - (mse / 255.0)
        
        return results

    def edge_case_detection(self, image):
        """Detect edge cases and non-property/vehicle content"""
        results = {
            "is_logo": False,
            "is_person_focused": False,
            "is_text_heavy": False,
            "is_abstract": False
        }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        if edge_density > 0.3:
            results["is_logo"] = True
        
        if self.yolo_model:
            try:
                yolo_results = self.yolo_model(image)
                person_area = 0
                total_area = image.shape[0] * image.shape[1]
                
                for result in yolo_results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        if class_name == 'person':
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            person_area += (x2 - x1) * (y2 - y1)
                
                if person_area / total_area > 0.3:
                    results["is_person_focused"] = True
            
            except Exception as e:
                logger.error(f"Edge case YOLO error: {e}")
        
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        
        high_freq_energy = np.mean(magnitude_spectrum)
        results["is_text_heavy"] = high_freq_energy > 130
        
        return results

    def calculate_property_score(self, validation_results):
        """Calculate overall property/vehicle relevance score with enhanced vehicle handling"""
        scores = []
        reasons = []
        
        obj_analysis = validation_results["object_analysis"]
        vehicle_analysis = validation_results["vehicle_specific_analysis"]
        
        # Object-based scoring
        object_score = 0
        vehicle_objects = obj_analysis["vehicle_objects"]
        property_objects = obj_analysis["property_objects"]
        
        valid_object_count = len(vehicle_objects) + len(property_objects)
        
        if valid_object_count > 0:
            # Higher weight for detected objects
            object_score += min(60, valid_object_count * 20)
            if vehicle_objects:
                vehicle_names = [obj['class'] for obj in vehicle_objects]
                reasons.append(f"Found {len(vehicle_objects)} vehicle objects: {vehicle_names}")
            if property_objects:
                property_names = [obj['class'] for obj in property_objects]
                reasons.append(f"Found {len(property_objects)} property objects: {property_names}")
        else:
            reasons.append("No valid property or vehicle objects detected")
        
        # Vehicle-specific scoring (even without YOLO detection)
        vehicle_specific_score = 0
        if vehicle_analysis["has_vehicle_characteristics"]:
            vehicle_specific_score += vehicle_analysis["vehicle_confidence"]
            reasons.append("Vehicle characteristics detected (wheels, metallic, symmetry)")
        
        # Combine object and vehicle analysis
        if vehicle_objects or vehicle_analysis["has_vehicle_characteristics"]:
            object_score = max(object_score, 40)  # Minimum score for vehicle images
            if not vehicle_objects and vehicle_analysis["has_vehicle_characteristics"]:
                reasons.append("Vehicle detected through visual analysis")
        
        # Penalty for forbidden objects
        forbidden_objects = len(obj_analysis["forbidden_objects"])
        if forbidden_objects > 0:
            object_score -= min(50, forbidden_objects * 25)
            reasons.append(f"Found {forbidden_objects} forbidden objects: {[obj['class'] for obj in obj_analysis['forbidden_objects']]}")
        
        scores.append(max(0, object_score + vehicle_specific_score))
        
        # Architectural analysis (reduced weight for vehicles)
        arch_analysis = validation_results["architectural_analysis"]
        arch_score = 0
        
        if arch_analysis["straight_lines"] > 10:
            arch_score += 5
        if arch_analysis["right_angles"] > 5:
            arch_score += 4
        if arch_analysis["structural_elements"] > 15:
            arch_score += 4
        
        # Reduce architectural weight for vehicle images
        if vehicle_objects or vehicle_analysis["has_vehicle_characteristics"]:
            arch_score = arch_score * 0.5  # Half weight for vehicles
        scores.append(arch_score)
        
        # Color analysis
        color_analysis = validation_results["color_analysis"]
        color_score = color_analysis["professional_score"] * 8
        
        # Boost for vehicle colors
        if vehicle_objects or vehicle_analysis["has_vehicle_characteristics"]:
            color_score += color_analysis["vehicle_color_score"]
            if color_analysis["is_vibrant"]:
                reasons.append("Vibrant vehicle color profile detected")
        else:
            if color_analysis["professional_score"] > 0.7:
                reasons.append("Professional color profile")
        
        scores.append(color_score)
        
        # Texture analysis
        texture_analysis = validation_results["texture_analysis"]
        texture_score = texture_analysis["structural_texture"] * 8
        
        # Boost for vehicle textures
        if vehicle_objects or vehicle_analysis["has_vehicle_characteristics"]:
            texture_score += texture_analysis["vehicle_texture_score"]
            if texture_analysis["is_metallic"]:
                reasons.append("Metallic vehicle texture detected")
        else:
            if texture_analysis["structural_texture"] > 0.1:
                reasons.append("Structural texture patterns detected")
        
        scores.append(texture_score)
        
        # Composition analysis
        comp_analysis = validation_results["composition_analysis"]
        comp_score = (comp_analysis["rule_of_thirds"] + comp_analysis["symmetry"]) * 3
        if comp_analysis["rule_of_thirds"] > 0.1:
            reasons.append("Good compositional balance")
        scores.append(comp_score)
        
        # Edge case penalties
        edge_cases = validation_results["edge_case_detection"]
        penalty = 0
        
        if edge_cases["is_logo"]:
            penalty += 8
            reasons.append("Logo-like image detected")
        
        if edge_cases["is_text_heavy"] and not (vehicle_objects or property_objects):
            penalty += 5
            reasons.append("Text-heavy image")
        
        if edge_cases["is_person_focused"]:
            penalty += 80  # Heavy penalty for person-focused images
            reasons.append("Person-focused image (potential NSFW)")
        
        penalty_score = max(0, -penalty)
        scores.append(penalty_score)
        
        # Final score calculation with balanced weights
        weights = [0.50, 0.08, 0.12, 0.12, 0.08, 0.10]  # Adjusted weights
        final_score = sum(score * weight for score, weight in zip(scores, weights))
        
        final_score = max(0, min(100, final_score))
        
        # Basic quality adjustments
        basic_checks = validation_results["basic_checks"]
        if basic_checks["min_dimension"] < 400:
            reasons.append("Image dimensions too small")
            final_score *= 0.8
        
        if basic_checks["blur_score"] < 50:
            reasons.append("Image appears blurry")
            final_score *= 0.9
        
        # Final vehicle boost if needed
        if (vehicle_objects or vehicle_analysis["has_vehicle_characteristics"]) and final_score < 50:
            boost = 35
            final_score = min(85, final_score + boost)
            reasons.append(f"Vehicle image boost applied (+{boost})")
        
        return final_score, reasons