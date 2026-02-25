"""
MediaPipe Hand Detection Module for AutoRig
Alternative to DWPose for hand refinement with better finger accuracy
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.request
import os

class MediaPipeHandDetector:
    """Wrapper for MediaPipe hand detection optimized for 3D hand reconstruction."""
    
    def __init__(self, static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe hand detector.
        
        Args:
            static_image_mode: Process each image independently (True for our use case)
            max_num_hands: Maximum number of hands to detect (2 for left+right)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        self.mp = mp
        
        # Download hand landmarker model if not exists
        model_path = self._ensure_model()
        
        # Create hand landmarker with the model file
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)

        
        # MediaPipe hand landmark indices (21 points, same as DWPose)
        self.LANDMARK_NAMES = {
            0: "WRIST",
            1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
            5: "INDEX_FINGER_MCP", 6: "INDEX_FINGER_PIP", 7: "INDEX_FINGER_DIP", 8: "INDEX_FINGER_TIP",
            9: "MIDDLE_FINGER_MCP", 10: "MIDDLE_FINGER_PIP", 11: "MIDDLE_FINGER_DIP", 12: "MIDDLE_FINGER_TIP",
            13: "RING_FINGER_MCP", 14: "RING_FINGER_PIP", 15: "RING_FINGER_DIP", 16: "RING_FINGER_TIP",
            17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP"
        }
    
    def _ensure_model(self):
        """Download hand landmarker model if not present."""
        # Store model in the same directory as this script
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "hand_landmarker.task"
        
        if not model_path.exists():
            print("### [MediaPipe] Downloading hand landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"### [MediaPipe] Model downloaded to {model_path}")
            except Exception as e:
                print(f"### [MediaPipe] Failed to download model: {e}")
                raise
        
        return model_path
    
    def detect_hands(self, image_path):
        """
        Detect hands in an image and return keypoints in DWPose-compatible format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Results in format matching DWPose output
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"### [MediaPipe] Failed to read image: {image_path}")
            return {'people': []}
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create MediaPipe Image object
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect hands
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.hand_landmarks or not detection_result.handedness:
            return {'people': []}
        
        # Initialize output
        left_hand = [0.0] * 63
        right_hand = [0.0] * 63
        left_detected = False
        right_detected = False
        
        # Process each detected hand
        # Note: We use the FIRST detected hand regardless of handedness label
        # because MediaPipe's handedness can be confused when viewing from different angles.
        # The calling code knows which hand it's rendering, so we just need the landmarks.
        for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
            # Get MediaPipe's handedness label for logging
            mp_label = handedness[0].category_name
            confidence = handedness[0].score
            
            # Extract landmarks to BOTH arrays - the caller will pick the right one
            # based on which hand type was requested
            for idx, landmark in enumerate(hand_landmarks):
                x = landmark.x * w
                y = landmark.y * h
                
                # Store in both arrays - caller knows which hand is being rendered
                left_hand[idx * 3] = x
                left_hand[idx * 3 + 1] = y
                left_hand[idx * 3 + 2] = confidence
                
                right_hand[idx * 3] = x
                right_hand[idx * 3 + 1] = y
                right_hand[idx * 3 + 2] = confidence
            
            left_detected = True
            right_detected = True
            
            # Only process the first detected hand
            break
        
        return {
            'people': [{
                'hand_left_keypoints_2d': left_hand if left_detected else [],
                'hand_right_keypoints_2d': right_hand if right_detected else []
            }]
        }
    
    def process_hand_images(self, image_paths, hand_type='left'):
        """
        Process multiple images of a hand and return all detections.
        
        Args:
            image_paths: List of image file paths
            hand_type: 'left' or 'right'
            
        Returns:
            list: List of detection results (one per image)
        """
        results = []
        detected_count = 0
        
        for i, img_path in enumerate(image_paths):
            detection = self.detect_hands(img_path)
            results.append(detection)
            
            # Check if target hand was detected
            if detection['people']:
                hand_key = f'hand_{hand_type}_keypoints_2d'
                if hand_key in detection['people'][0] and len(detection['people'][0][hand_key]) > 0:
                    detected_count += 1
        

        return results
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()


def convert_mediapipe_to_autorig_format(mediapipe_results):
    """
    Convert MediaPipe hand detection results to AutoRig's expected format.
    
    MediaPipe landmarks match OpenPose/DWPose 21-point hand format:
    - Index 0: Wrist
    - Indices 1-4: Thumb (CMC, MCP, IP, TIP)
    - Indices 5-8: Index finger (MCP, PIP, DIP, TIP)
    - Indices 9-12: Middle finger
    - Indices 13-16: Ring finger
    - Indices 17-20: Pinky finger
    
    Args:
        mediapipe_results: Output from MediaPipeHandDetector.detect_hands()
        
    Returns:
        dict: Same format (already compatible)
    """
    # MediaPipe output is already in DWPose-compatible format
    return mediapipe_results


# Example usage for testing
if __name__ == "__main__":
    detector = MediaPipeHandDetector()
    
    # Test single image
    test_image = "test_hand.png"
    results = detector.detect_hands(test_image)
    print(f"Detected hands: {len(results['people'])}")
    
    if results['people']:
        person = results['people'][0]
        if person.get('hand_left_keypoints_2d'):
            print("Left hand detected")
        if person.get('hand_right_keypoints_2d'):
            print("Right hand detected")
    
    detector.close()
