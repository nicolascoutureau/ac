import time
import cv2
import scenedetect
import subprocess
import shutil
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os
import numpy as np
from tqdm import tqdm
import threading
import queue
import urllib.request
import tempfile
import sys

# Version tracking for Replicate deployment
AUTOCROP_VERSION = "0.18"

# ============= Fast Speaker Detection (MediaPipe) =============
# Uses MediaPipe Face Mesh for real-time lip tracking + WebRTC VAD for audio

MEDIAPIPE_AVAILABLE = False
FACE_MESH = None

def init_mediapipe():
    """Initialize MediaPipe Face Mesh for fast lip tracking."""
    global MEDIAPIPE_AVAILABLE, FACE_MESH
    try:
        # Try modern MediaPipe API first
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            from mediapipe import solutions
            
            FACE_MESH = solutions.face_mesh.FaceMesh(
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            MEDIAPIPE_AVAILABLE = True
            print("  ✓ MediaPipe Face Mesh initialized (solutions API)")
            return True
        except (ImportError, AttributeError):
            pass
        
        # Try legacy API
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=4,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                MEDIAPIPE_AVAILABLE = True
                print("  ✓ MediaPipe Face Mesh initialized (legacy API)")
                return True
        except (ImportError, AttributeError):
            pass
        
        # Try direct import
        try:
            from mediapipe.python.solutions import face_mesh
            FACE_MESH = face_mesh.FaceMesh(
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            MEDIAPIPE_AVAILABLE = True
            print("  ✓ MediaPipe Face Mesh initialized (direct import)")
            return True
        except (ImportError, AttributeError):
            pass
        
        print("  ⚠️ MediaPipe Face Mesh not available in this version")
        MEDIAPIPE_AVAILABLE = False
        return False
        
    except ImportError:
        print("  ⚠️ MediaPipe not installed")
        MEDIAPIPE_AVAILABLE = False
        return False
    except Exception as e:
        print(f"  ⚠️ MediaPipe init failed: {e}")
        MEDIAPIPE_AVAILABLE = False
        return False


def get_mouth_aperture(face_landmarks, frame_height):
    """
    Calculate mouth aperture (openness) from MediaPipe face landmarks.
    Returns normalized value 0-1 where higher = more open mouth.
    """
    # MediaPipe lip landmarks
    # Upper lip: 13 (center top), Lower lip: 14 (center bottom)
    # Outer corners: 61 (left), 291 (right)
    try:
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[291]
        
        # Vertical mouth opening
        mouth_height = abs(lower_lip.y - upper_lip.y) * frame_height
        
        # Horizontal mouth width (for normalization)
        mouth_width = abs(right_corner.x - left_corner.x) * frame_height
        
        if mouth_width > 0:
            # Normalized aperture ratio
            return mouth_height / mouth_width
        return 0.0
    except:
        return 0.0


def get_face_orientation_score(face_landmarks):
    """
    Calculate how front-facing a face is based on MediaPipe landmarks.
    Returns score 0-1 where 1 = perfectly front-facing, 0 = profile.
    
    Uses nose position relative to face width - if nose is centered, face is front-facing.
    Also checks if both eyes are visible and symmetric.
    """
    try:
        # Key landmarks
        nose_tip = face_landmarks.landmark[1]  # Nose tip
        left_eye_outer = face_landmarks.landmark[33]  # Left eye outer corner
        right_eye_outer = face_landmarks.landmark[263]  # Right eye outer corner
        left_cheek = face_landmarks.landmark[234]  # Left face edge
        right_cheek = face_landmarks.landmark[454]  # Right face edge
        
        # Face width (cheek to cheek)
        face_width = abs(right_cheek.x - left_cheek.x)
        if face_width < 0.01:
            return 0.0
        
        # Face center X
        face_center_x = (left_cheek.x + right_cheek.x) / 2
        
        # How centered is the nose? (0 = perfectly centered)
        nose_offset = abs(nose_tip.x - face_center_x) / face_width
        
        # Eye symmetry - both eyes should be roughly equidistant from center
        left_eye_dist = abs(left_eye_outer.x - face_center_x)
        right_eye_dist = abs(right_eye_outer.x - face_center_x)
        eye_asymmetry = abs(left_eye_dist - right_eye_dist) / face_width
        
        # Combined score: penalize off-center nose and asymmetric eyes
        # Perfect front-facing: nose_offset ≈ 0, eye_asymmetry ≈ 0
        orientation_score = max(0, 1.0 - (nose_offset * 3) - (eye_asymmetry * 2))
        
        return orientation_score
    except:
        return 0.5  # Default to neutral if landmarks unavailable


def detect_audio_activity(video_path, start_frame, end_frame, fps):
    """
    Detect audio activity in a video segment using WebRTC VAD.
    Returns list of (frame_start, frame_end, is_speech) segments.
    """
    try:
        import webrtcvad
        import wave
        import struct
        
        # Extract audio segment
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Extract audio with ffmpeg
        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration),
            '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', tmp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Read audio and run VAD
        vad = webrtcvad.Vad(2)  # Aggressiveness 0-3, 2 is balanced
        
        with wave.open(tmp_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
        
        os.unlink(tmp_path)
        
        # Process in 30ms frames (480 samples at 16kHz)
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        speech_frames = []
        for i in range(0, len(audio_data) - frame_size * 2, frame_size * 2):
            frame = audio_data[i:i + frame_size * 2]
            if len(frame) == frame_size * 2:
                is_speech = vad.is_speech(frame, sample_rate)
                # Convert audio frame to video frame
                audio_time = i / (sample_rate * 2)
                video_frame = int(start_frame + audio_time * fps)
                speech_frames.append((video_frame, is_speech))
        
        return speech_frames
        
    except Exception as e:
        # Fallback: assume speech throughout
        return [(start_frame, True)]


def detect_speaker_fast(video_path, start_frame, end_frame, people_detections, fps, sample_interval=5, max_people=6):
    """
    Fast speaker detection using MediaPipe lip tracking on FULL FRAMES.
    Runs MediaPipe on entire frame to detect all faces fresh (not using pre-computed boxes).
    Then matches the speaking face back to people_detections.
    
    Args:
        video_path: Path to video
        start_frame, end_frame: Frame range to analyze
        people_detections: List of detected people with face_box
        fps: Video frame rate
        sample_interval: Sample every N frames for efficiency
        max_people: Max people to analyze (default: 6)
        
    Returns:
        (speaker_index, is_conversation, []) - speaker_index is index into people_detections, None if undetermined
    """
    if not MEDIAPIPE_AVAILABLE or FACE_MESH is None:
        return None, False, []
    
    if len(people_detections) == 0:
        return None, False, []
    
    if len(people_detections) == 1:
        # Single person - assume they're speaking
        return 0, False, []
    
    # Multiple people - track lip movement AND face orientation using MediaPipe on full frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, False, []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Track mouth aperture and orientation for each face (keyed by approximate X position)
    # We use X position buckets to track faces across frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_buckets = 10  # Divide frame into 10 horizontal zones
    bucket_width = frame_width / num_buckets
    aperture_history = {i: [] for i in range(num_buckets)}
    orientation_history = {i: [] for i in range(num_buckets)}  # Track face orientation
    
    frame_count = 0
    frames_analyzed = 0
    max_frames = min(90, end_frame - start_frame)  # Limit to ~3 seconds at 30fps
    
    try:
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < min(end_frame, start_frame + max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_interval != 0:
                continue
            
            frames_analyzed += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            
            # Run MediaPipe on FULL FRAME to detect all faces
            results = FACE_MESH.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get face center X position from nose tip (landmark 1)
                    nose_x = face_landmarks.landmark[1].x * frame_width
                    bucket = min(int(nose_x / bucket_width), num_buckets - 1)
                    
                    # Calculate mouth aperture
                    aperture = get_mouth_aperture(face_landmarks, frame_height)
                    aperture_history[bucket].append(aperture)
                    
                    # Calculate face orientation (front-facing score)
                    orientation = get_face_orientation_score(face_landmarks)
                    orientation_history[bucket].append(orientation)
    
    finally:
        cap.release()
    
    # Calculate combined scores: lip movement weighted by face orientation
    # Front-facing speakers are prioritized
    combined_scores = {}
    for bucket, apertures in aperture_history.items():
        if len(apertures) > 3:
            # Lip movement score
            variance = np.var(apertures)
            max_change = max(abs(apertures[i] - apertures[i-1]) 
                           for i in range(1, len(apertures))) if len(apertures) > 1 else 0
            lip_score = variance + max_change * 0.5
            
            # Average orientation (how front-facing)
            avg_orientation = np.mean(orientation_history[bucket]) if orientation_history[bucket] else 0.5
            
            # Combined score: lip movement * orientation boost
            # Front-facing (orientation=1.0) gets full score
            # Profile (orientation=0.3) gets reduced score
            orientation_boost = 0.5 + avg_orientation * 0.5  # Range: 0.5 to 1.0
            combined_scores[bucket] = (lip_score, lip_score * orientation_boost, avg_orientation)
    
    if not combined_scores:
        return None, False, []
    
    # Find bucket with best combined score
    sorted_buckets = sorted(combined_scores.items(), key=lambda x: x[1][1], reverse=True)  # Sort by boosted score
    top_bucket, (raw_score, boosted_score, orientation) = sorted_buckets[0]
    speaker_x = (top_bucket + 0.5) * bucket_width  # Center of speaking bucket
    
    # Log analysis
    total_samples = sum(len(v) for v in aperture_history.values())
    facing = "front" if orientation > 0.7 else ("side" if orientation > 0.4 else "profile")
    print(f"    📊 MediaPipe: {frames_analyzed} frames, {total_samples} samples, speaker at X≈{int(speaker_x)} ({facing}-facing, score:{boosted_score:.4f})")
    
    if boosted_score < 0.0001:
        print(f"    📊 No lip movement detected (score: {boosted_score:.6f})")
        # No clear speaker - if multiple front-facing people, use letterbox
        front_facing_count = sum(1 for _, (_, _, o) in combined_scores.items() if o > 0.5)
        if front_facing_count >= 2:
            print(f"    📊 No speaker but {front_facing_count} front-facing people → letterbox")
            return None, True, []
        return None, False, []
    
    # Check for conversation (multiple active front-facing speakers)
    if len(sorted_buckets) > 1:
        _, (_, second_boosted, second_orient) = sorted_buckets[1]
        ratio = boosted_score / second_boosted if second_boosted > 0 else float('inf')
        
        # Debug: show why conversation was/wasn't detected
        if second_boosted > 0:
            print(f"    📊 2nd speaker: score={second_boosted:.4f}, orient={second_orient:.2f}, ratio={ratio:.2f}")
        
        # Conversation if:
        # - Both have similar lip movement (ratio < 2.5)
        # - Second person is at least side-facing (orient > 0.3)
        if second_boosted > 0.00005 and ratio < 2.5 and second_orient > 0.3:
            print(f"    📊 Conversation detected → letterbox")
            return None, True, []
    
    # Match speaking position to people_detections
    # Find person whose face/person box is closest to speaker_x
    best_match = None
    best_distance = float('inf')
    
    for i, person in enumerate(people_detections):
        # Use face_box if available, otherwise person_box
        box = person.get('face_box') or person.get('person_box')
        if box:
            person_center_x = (box[0] + box[2]) / 2
            distance = abs(person_center_x - speaker_x)
            if distance < best_distance:
                best_distance = distance
                best_match = i
    
    if best_match is not None:
        print(f"    📊 Speaker matched to person {best_match + 1} (distance: {int(best_distance)}px)")
        return best_match, False, []
    
    return None, False, []


def load_yolo_face_detector():
    """
    Load YOLO face detector model if available.
    Falls back to face estimation from person boxes if no face model found.
    
    Returns:
        YOLO model for face detection, or None to use estimation
    """
    try:
        from ultralytics import YOLO
        import torch
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for YOLO face model in common locations
        face_model_paths = [
            os.path.join(script_dir, "yolov8n-face.pt"),  # Same dir as autocrop.py
            "/src/yolov8n-face.pt",  # Replicate container /src/
            "yolov8n-face.pt",  # Current working directory
            os.path.join(script_dir, "yolov8s-face.pt"),
            os.path.join(script_dir, "models", "yolov8n-face.pt"),
        ]
        
        for model_path in face_model_paths:
            if os.path.exists(model_path):
                print(f"✅ Loading YOLO face model: {model_path}")
                model = YOLO(model_path)
                # Move to GPU if available
                if torch.cuda.is_available():
                    model.to('cuda')
                    print(f"  🚀 Face model on GPU: {torch.cuda.get_device_name(0)}")
                return model
        
        # No dedicated face model found - will use estimation from person boxes
        print("ℹ️  No YOLO face model found, using face estimation from person boxes")
        return None
        
    except Exception as e:
        print(f"⚠️  Could not load YOLO face detector: {e}")
        return None


def estimate_face_from_person(person_box, frame_height, frame_width):
    """
    Estimate face bounding box from person bounding box.
    Works well for talking head / upper body shots.
    
    Args:
        person_box: Person bounding box [x1, y1, x2, y2]
        frame_height: Frame height for validation
        frame_width: Frame width for validation
        
    Returns:
        Tuple of (face_box, confidence) or (None, 0) if estimation fails
    """
    x1, y1, x2, y2 = person_box
    person_width = x2 - x1
    person_height = y2 - y1
    
    if person_height <= 0 or person_width <= 0:
        return None, 0.0
    
    # Estimate face position based on person box
    # Face is typically in the upper 25-35% of person box, centered horizontally
    
    # Calculate aspect ratio to determine shot type
    aspect = person_height / person_width if person_width > 0 else 1.0
    
    if aspect > 2.5:
        # Full body shot - face is small, in upper 15-20%
        face_top_ratio = 0.02
        face_bottom_ratio = 0.18
        face_width_ratio = 0.4
        confidence = 0.6  # Lower confidence for full body
    elif aspect > 1.5:
        # Medium shot - face in upper 25-35%
        face_top_ratio = 0.02
        face_bottom_ratio = 0.30
        face_width_ratio = 0.5
        confidence = 0.75
    else:
        # Close-up / head shot - face fills more of the box
        face_top_ratio = 0.05
        face_bottom_ratio = 0.55
        face_width_ratio = 0.7
        confidence = 0.85
    
    # Calculate face box
    face_height = person_height * (face_bottom_ratio - face_top_ratio)
    face_width = min(person_width * face_width_ratio, face_height * 0.8)  # Face aspect ~0.7-0.8
    
    face_center_x = (x1 + x2) / 2
    face_top = y1 + person_height * face_top_ratio
    
    face_x1 = int(face_center_x - face_width / 2)
    face_y1 = int(face_top)
    face_x2 = int(face_center_x + face_width / 2)
    face_y2 = int(face_top + face_height)
    
    # Clamp to frame bounds
    face_x1 = max(0, min(face_x1, frame_width - 1))
    face_y1 = max(0, min(face_y1, frame_height - 1))
    face_x2 = max(face_x1 + 1, min(face_x2, frame_width))
    face_y2 = max(face_y1 + 1, min(face_y2, frame_height))
    
    face_box = [face_x1, face_y1, face_x2, face_y2]
    
    return face_box, confidence


# Keep old function name for compatibility but redirect to new implementation
def load_dnn_face_detector():
    """
    Legacy function - now loads YOLO face detector or returns None for estimation.
    """
    return load_yolo_face_detector()

def compute_saliency_region(frame, aspect_ratio, padding_factor=1.5):
    """
    Compute the most visually interesting region in a frame.
    Uses edge detection and gradient magnitude (works without opencv-contrib).
    
    Args:
        frame: Input frame (BGR)
        aspect_ratio: Target aspect ratio (width/height)
        padding_factor: How much padding around salient region (1.5 = 50% padding)
        
    Returns:
        Bounding box [x1, y1, x2, y2] of the most interesting region, or None if failed
    """
    try:
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute gradient magnitude using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255
        gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
        
        # Apply Gaussian blur to smooth
        gradient_mag = cv2.GaussianBlur(gradient_mag, (31, 31), 0)
        
        # Threshold to find high-detail regions (top 25% most detailed)
        threshold = np.percentile(gradient_mag, 75)
        _, binary_map = cv2.threshold(gradient_mag, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of interesting regions
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No distinct region - use weighted center of gradient magnitude
            moments = cv2.moments(gradient_mag)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                # Fallback to center
                cx, cy = frame_width // 2, frame_height // 2
        else:
            # Get bounding box of all interesting contours combined
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            cx = x + w // 2
            cy = y + h // 2
        
        # Create crop region centered on the interesting area
        crop_height = frame_height
        crop_width = int(crop_height * aspect_ratio)
        
        # Constrain to frame
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = int(crop_width / aspect_ratio)
        
        # Center on the interesting point
        x1 = cx - crop_width // 2
        y1 = cy - crop_height // 2
        
        # Shift to keep within bounds
        if x1 < 0:
            x1 = 0
        elif x1 + crop_width > frame_width:
            x1 = frame_width - crop_width
        
        if y1 < 0:
            y1 = 0
        elif y1 + crop_height > frame_height:
            y1 = frame_height - crop_height
        
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        
        return [int(x1), int(y1), int(x2), int(y2)]
        
    except Exception as e:
        print(f"  ⚠️  Interest detection failed: {e}")
        return None


def get_center_crop_box(frame_width, frame_height, aspect_ratio):
    """
    Get a center crop box for the given aspect ratio.
    
    Args:
        frame_width: Frame width
        frame_height: Frame height
        aspect_ratio: Target aspect ratio (width/height)
        
    Returns:
        Bounding box [x1, y1, x2, y2] for center crop
    """
    target_width = int(frame_height * aspect_ratio)
    
    if target_width <= frame_width:
        # Crop width, keep full height
        x1 = (frame_width - target_width) // 2
        return [x1, 0, x1 + target_width, frame_height]
    else:
        # Crop height, keep full width
        target_height = int(frame_width / aspect_ratio)
        y1 = (frame_height - target_height) // 2
        return [0, y1, frame_width, y1 + target_height]


def detect_face_yolo(frame, face_model, person_box=None, min_confidence=0.5):
    """
    Detect faces using YOLO face model with confidence scores.
    
    Args:
        frame: Input frame (BGR)
        face_model: YOLO face detection model
        person_box: Optional [x1, y1, x2, y2] to restrict search area
        min_confidence: Minimum confidence threshold for face detection
        
    Returns:
        List of (face_box, confidence) tuples
    """
    if person_box:
        x1, y1, x2, y2 = person_box
        roi = frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        roi = frame
        offset_x, offset_y = 0, 0
    
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return []
    
    # Run YOLO face detection
    results = face_model([roi], verbose=False)
    
    faces = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence > min_confidence:
                fx1, fy1, fx2, fy2 = [int(i) for i in box.xyxy[0]]
                
                # Offset back to full frame coordinates
                face_box = [
                    max(0, fx1 + offset_x),
                    max(0, fy1 + offset_y),
                    fx2 + offset_x,
                    fy2 + offset_y
                ]
                faces.append((face_box, float(confidence)))
    
    return faces


# Keep old function name for compatibility
def detect_face_dnn(frame, face_detector, person_box=None, min_confidence=0.5):
    """
    Unified face detection - uses YOLO if model provided, otherwise estimates from person box.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # If we have a YOLO face model, use it
    if face_detector is not None:
        try:
            return detect_face_yolo(frame, face_detector, person_box, min_confidence)
        except:
            pass
    
    # Fallback: estimate face from person box
    if person_box is not None:
        face_box, confidence = estimate_face_from_person(person_box, frame_height, frame_width)
        if face_box is not None and confidence >= min_confidence:
            return [(face_box, confidence)]
    
    return []


def analyze_single_frame(frame, model, face_cascade, dnn_face_detector, analysis_scale, confidence_threshold):
    """
    Analyze a single frame for people and faces.
    
    Returns:
        List of detected objects with person_box, face_box, confidence, face_confidence
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Resize frame for faster analysis if scale < 1.0
    if analysis_scale < 1.0:
        analysis_width = int(frame_width * analysis_scale)
        analysis_height = int(frame_height * analysis_scale)
        analysis_frame = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_LINEAR)
        scale_back_x = frame_width / analysis_width
        scale_back_y = frame_height / analysis_height
    else:
        analysis_frame = frame
        scale_back_x = scale_back_y = 1.0
    
    results = model([analysis_frame], verbose=False)
    
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:  # Person class
                confidence = float(box.conf[0])
                
                if confidence < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                
                # Scale bounding box back to original resolution
                x1 = int(x1 * scale_back_x)
                y1 = int(y1 * scale_back_y)
                x2 = int(x2 * scale_back_x)
                y2 = int(y2 * scale_back_y)
                
                # Clamp to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width, x2)
                y2 = min(frame_height, y2)
                
                person_box = [x1, y1, x2, y2]
                
                # Detect face within person box
                face_box = None
                face_confidence = 0.0
                
                if dnn_face_detector is not None:
                    # Use YOLO face model if available
                    faces = detect_face_dnn(frame, dnn_face_detector, person_box, min_confidence=0.5)
                    if faces:
                        faces.sort(key=lambda x: x[1], reverse=True)
                        face_box, face_confidence = faces[0]
                else:
                    # Estimate face from person bounding box (faster, no extra model needed)
                    face_box, face_confidence = estimate_face_from_person(
                        person_box, frame_height, frame_width
                    )
                    
                    # If estimation confidence is low, try Haar cascade as fallback
                    if face_confidence < 0.7 and face_cascade is not None:
                        try:
                            person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            haar_faces = face_cascade.detectMultiScale(
                                person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                            )
                            if len(haar_faces) > 0:
                                fx, fy, fw, fh = haar_faces[0]
                                face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]
                                face_confidence = 0.8
                        except:
                            pass  # Keep estimation result

                detected_objects.append({
                    'person_box': person_box, 
                    'face_box': face_box,
                    'confidence': confidence,
                    'face_confidence': face_confidence
                })
    
    # Fallback: face detection if no people found
    if len(detected_objects) == 0:
        if dnn_face_detector is not None:
            faces = detect_face_dnn(frame, dnn_face_detector, min_confidence=0.4)
            for face_box, face_conf in faces:
                fx1, fy1, fx2, fy2 = face_box
                fw, fh = fx2 - fx1, fy2 - fy1
                estimated_body_height = fh * 5
                padding_x = int(fw * 0.8)
                person_box = [
                    max(0, fx1 - padding_x),
                    max(0, fy1 - int(fh * 0.3)),
                    min(frame_width, fx2 + padding_x),
                    min(frame_height, fy1 + estimated_body_height)
                ]
                detected_objects.append({
                    'person_box': person_box, 
                    'face_box': face_box,
                    'confidence': face_conf * 0.7,
                    'face_confidence': face_conf
                })
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_box = [x, y, x + w, y + h]
                estimated_body_height = h * 5
                padding_x = int(w * 0.8)
                person_box = [
                    max(0, x - padding_x),
                    max(0, y - int(h * 0.3)),
                    min(frame_width, x + w + padding_x),
                    min(frame_height, y + estimated_body_height)
                ]
                detected_objects.append({
                    'person_box': person_box, 
                    'face_box': face_box,
                    'confidence': 0.4,
                    'face_confidence': 0.7
                })
    
    return detected_objects


def merge_detections_across_frames(all_frame_detections, iou_threshold=0.5):
    """
    Merge detections from multiple frames into a single best detection list.
    Uses IoU (Intersection over Union) to match people across frames.
    
    Args:
        all_frame_detections: List of detection lists, one per frame
        iou_threshold: Minimum IoU to consider same person
        
    Returns:
        Merged list with averaged/best positions and boosted confidence
    """
    if not all_frame_detections:
        return []
    
    # Flatten all detections with frame index
    all_detections = []
    for frame_idx, detections in enumerate(all_frame_detections):
        for det in detections:
            all_detections.append({**det, 'frame_idx': frame_idx})
    
    if not all_detections:
        return []
    
    # Group detections by person using IoU matching
    person_groups = []
    used = set()
    
    for i, det1 in enumerate(all_detections):
        if i in used:
            continue
        
        group = [det1]
        used.add(i)
        
        for j, det2 in enumerate(all_detections):
            if j in used:
                continue
            
            # Calculate IoU between person boxes
            box1 = det1['person_box']
            box2 = det2['person_box']
            
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                iou = intersection / (area1 + area2 - intersection)
                
                if iou > iou_threshold:
                    group.append(det2)
                    used.add(j)
        
        person_groups.append(group)
    
    # Merge each group into single detection
    merged_detections = []
    num_frames = len(all_frame_detections)
    
    for group in person_groups:
        # Average person box coordinates
        avg_person_box = [
            int(sum(d['person_box'][0] for d in group) / len(group)),
            int(sum(d['person_box'][1] for d in group) / len(group)),
            int(sum(d['person_box'][2] for d in group) / len(group)),
            int(sum(d['person_box'][3] for d in group) / len(group))
        ]
        
        # Average face box (only from detections that have faces)
        faces_with_box = [d for d in group if d['face_box'] is not None]
        if faces_with_box:
            avg_face_box = [
                int(sum(d['face_box'][0] for d in faces_with_box) / len(faces_with_box)),
                int(sum(d['face_box'][1] for d in faces_with_box) / len(faces_with_box)),
                int(sum(d['face_box'][2] for d in faces_with_box) / len(faces_with_box)),
                int(sum(d['face_box'][3] for d in faces_with_box) / len(faces_with_box))
            ]
            avg_face_confidence = sum(d['face_confidence'] for d in faces_with_box) / len(faces_with_box)
        else:
            avg_face_box = None
            avg_face_confidence = 0.0
        
        # Boost confidence based on how many frames the person was detected in
        frames_detected = len(set(d['frame_idx'] for d in group))
        detection_consistency = frames_detected / num_frames
        
        # Average confidence, boosted by consistency
        avg_confidence = sum(d['confidence'] for d in group) / len(group)
        boosted_confidence = min(1.0, avg_confidence * (1 + detection_consistency * 0.3))
        
        merged_detections.append({
            'person_box': avg_person_box,
            'face_box': avg_face_box,
            'confidence': boosted_confidence,
            'face_confidence': avg_face_confidence,
            'frames_detected': frames_detected,
            'total_frames': num_frames
        })
    
    # Sort by confidence
    merged_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return merged_detections


def compute_scene_fallback(video_path, scene_start_time, scene_end_time, aspect_ratio, fallback_strategy='saliency'):
    """
    Compute a fallback crop region for scenes without people.
    Uses saliency detection or center crop.
    
    Args:
        video_path: Path to video
        scene_start_time: Scene start time
        scene_end_time: Scene end time
        aspect_ratio: Target aspect ratio
        fallback_strategy: 'saliency', 'center', or 'letterbox'
        
    Returns:
        Bounding box [x1, y1, x2, y2] for the fallback crop, or None for letterbox
    """
    if fallback_strategy == 'letterbox':
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Get middle frame of scene
    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame = int(start_frame + (end_frame - start_frame) / 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    frame_height, frame_width = frame.shape[:2]
    
    if fallback_strategy == 'saliency':
        # Try saliency detection
        saliency_box = compute_saliency_region(frame, aspect_ratio)
        if saliency_box:
            print(f"  🎯 Using saliency detection (no people found)")
            return saliency_box
        # Fall through to center if saliency fails
    
    # Center crop fallback
    print(f"  📍 Using center crop (no people found)")
    return get_center_crop_box(frame_width, frame_height, aspect_ratio)


def analyze_scene_content(video_path, scene_start_time, scene_end_time, model, face_cascade, analysis_scale=1.0, dnn_face_detector=None, confidence_threshold=0.3, num_sample_frames=3):
    """
    Analyzes multiple frames of a scene to detect people and faces.
    Uses multi-frame sampling for more reliable detection.
    
    Args:
        confidence_threshold: Minimum confidence (0-1) for YOLO person detection. Default: 0.3
        dnn_face_detector: Optional OpenCV DNN face detector for confidence scores
        analysis_scale: Scale factor for analysis (e.g., 0.5 = half resolution)
        num_sample_frames: Number of frames to sample from the scene (default: 3)
        
    Returns:
        List of merged detections with averaged positions and boosted confidence
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    scene_duration = end_frame - start_frame
    
    # Determine sample frame positions
    # For short scenes, use fewer samples
    if scene_duration < 10:
        num_sample_frames = 1
    elif scene_duration < 30:
        num_sample_frames = min(2, num_sample_frames)
    else:
        num_sample_frames = min(num_sample_frames, 5)  # Cap at 5 for performance
    
    # Calculate frame positions to sample (evenly distributed)
    if num_sample_frames == 1:
        sample_positions = [int(start_frame + scene_duration / 2)]
    else:
        # Sample at 20%, 50%, 80% of scene (avoid very start/end)
        step = scene_duration / (num_sample_frames + 1)
        sample_positions = [int(start_frame + step * (i + 1)) for i in range(num_sample_frames)]
    
    # Collect detections from each sample frame
    all_frame_detections = []
    fallback_used = False
    
    for frame_pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        detections = analyze_single_frame(
            frame, model, face_cascade, dnn_face_detector, 
            analysis_scale, confidence_threshold
        )
        
        if detections:
            all_frame_detections.append(detections)
        elif not fallback_used:
            # Try with lower threshold on this frame
            detections = analyze_single_frame(
                frame, model, face_cascade, dnn_face_detector,
                analysis_scale, confidence_threshold * 0.5
            )
            if detections:
                all_frame_detections.append(detections)
                fallback_used = True
    
    cap.release()
    
    # If no detections from any frame, try one more time with very low threshold
    if not all_frame_detections:
        # Last attempt with very low confidence
        cap = cv2.VideoCapture(video_path)
        middle_frame = int(start_frame + scene_duration / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            detections = analyze_single_frame(
                frame, model, face_cascade, dnn_face_detector,
                analysis_scale, confidence_threshold * 0.3  # Very low threshold
            )
            if detections:
                print(f"  ⚠️  Found people with low confidence ({confidence_threshold * 0.3:.2f})")
                return detections
        
        print("  ⚠️  No people detected in any sampled frame")
        return []
    
    # Merge detections across frames
    merged = merge_detections_across_frames(all_frame_detections)
    
    # Log multi-frame analysis results
    if len(sample_positions) > 1 and merged:
        consistent_people = sum(1 for d in merged if d.get('frames_detected', 1) > 1)
        if consistent_people > 0:
            pass  # Detected consistently across frames - good!
    
    return merged


def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    
    # If no scene cuts detected, treat the entire video as a single scene
    if not scene_list:
        base_timecode = video_manager.get_base_timecode()
        end_timecode = video_manager.get_current_timecode()
        scene_list = [(base_timecode, end_timecode)]
        print("  ℹ️  No scene cuts detected - treating entire video as one scene")
    
    video_manager.release()
    return scene_list, fps


def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]


def calculate_box_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Used to detect if two detections are likely the same person.
    
    Args:
        box1, box2: Bounding boxes [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def would_require_excessive_zoom(target_box, frame_width, frame_height, aspect_ratio, max_zoom=4.0):
    """
    Check if tracking this target would require excessive zoom.
    Very lenient - almost always allows tracking over letterbox.
    
    Args:
        target_box: Bounding box [x1, y1, x2, y2]
        frame_width: Frame width
        frame_height: Frame height
        aspect_ratio: Target aspect ratio
        max_zoom: Maximum acceptable zoom factor (default: 4.0)
        
    Returns:
        True if zoom would exceed max_zoom (rarely returns True now)
    """
    target_height = target_box[3] - target_box[1]
    target_width = target_box[2] - target_box[0]
    
    if target_height <= 0 or target_width <= 0:
        return True
    
    # Very lenient: only reject if target is tiny (< 5% of frame height)
    # This almost never triggers letterbox when something is detected
    if target_height / frame_height < 0.05:
        return True
    
    return False


def calculate_overlap_ratio(box1, box2):
    """
    Calculate the overlap ratio between two boxes.
    Returns the intersection area divided by the smaller box's area.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0  # No overlap
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    smaller_area = min(area1, area2)
    if smaller_area <= 0:
        return 0.0
    
    return intersection / smaller_area


def are_people_too_close_for_split(person1, person2, frame_width, frame_height, aspect_ratio):
    """
    Check if two people are too close together for split-screen to work well.
    If their crops would overlap significantly, split-screen won't look good.
    
    Returns True if people are too close (should track one instead of split)
    """
    box1 = person1['person_box']
    box2 = person2['person_box']
    
    # Check direct overlap of person boxes
    direct_overlap = calculate_overlap_ratio(box1, box2)
    if direct_overlap > 0.3:  # More than 30% overlap
        return True
    
    # Check horizontal separation - for split to work, people need horizontal space
    center1_x = (box1[0] + box1[2]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    
    # Calculate how wide each person's crop would be
    # For split-screen, each person gets a crop that's approximately:
    # crop_width = person_height * 1.5 * section_aspect_ratio (rough estimate)
    person1_height = box1[3] - box1[1]
    person2_height = box2[3] - box2[1]
    
    # Section aspect for vertical stacking (9:16 split in half vertically)
    if aspect_ratio < 0.8:  # Vertical output
        section_aspect = aspect_ratio * 2  # Each section is wider relative to its height
    else:
        section_aspect = aspect_ratio / 2
    
    # Estimated crop width for each person (to fill ~50% of crop)
    crop_width1 = person1_height * 1.8 * section_aspect
    crop_width2 = person2_height * 1.8 * section_aspect
    
    # Check if crops would overlap significantly
    crop1_left = center1_x - crop_width1 / 2
    crop1_right = center1_x + crop_width1 / 2
    crop2_left = center2_x - crop_width2 / 2
    crop2_right = center2_x + crop_width2 / 2
    
    # Calculate overlap of estimated crop regions
    overlap_left = max(crop1_left, crop2_left)
    overlap_right = min(crop1_right, crop2_right)
    
    if overlap_right > overlap_left:
        overlap_width = overlap_right - overlap_left
        smaller_crop_width = min(crop_width1, crop_width2)
        crop_overlap_ratio = overlap_width / smaller_crop_width if smaller_crop_width > 0 else 0
        
        if crop_overlap_ratio > 0.5:  # More than 50% crop overlap
            return True
    
    return False


def decide_cropping_strategy(scene_analysis, frame_height, frame_width, aspect_ratio, max_zoom=4.0, active_speaker_idx=None):
    """
    Decide the cropping strategy based on scene content.
    Falls back to tracking single person when split-screen would have too much overlap.
    Prioritizes active speaker when detected.
    
    Args:
        max_zoom: Maximum acceptable zoom factor before falling back to letterbox (default: 4.0)
        active_speaker_idx: Index of active speaker to focus on (if detected)
    """
    num_people = len(scene_analysis)
    if num_people == 0:
        print(f"  ⚠️  No people detected in scene, will use fallback")
        return 'LETTERBOX', None
    
    if num_people == 1:
        person = scene_analysis[0]
        # Use person_box for tracking (better framing), face_box only influences centering
        target_box = person['person_box']
        
        # Check if zoom would be excessive for single person
        if would_require_excessive_zoom(target_box, frame_width, frame_height, aspect_ratio, max_zoom):
            print(f"  ⚠️  Person too small, using LETTERBOX")
            return 'LETTERBOX', None
        
        return 'TRACK', target_box
    
    # If we have an active speaker, focus on them
    if active_speaker_idx is not None and 0 <= active_speaker_idx < num_people:
        speaker = scene_analysis[active_speaker_idx]
        speaker_box = speaker['person_box']
        speaker_conf = speaker.get('confidence', 0)
        
        if not would_require_excessive_zoom(speaker_box, frame_width, frame_height, aspect_ratio, max_zoom):
            print(f"  🎤 Speaker detected (conf: {speaker_conf:.2f}) - tracking")
            return 'TRACK', speaker_box
    
    # Multiple people detected - sort by confidence and take top detections
    sorted_by_confidence = sorted(scene_analysis, key=lambda x: x['confidence'], reverse=True)
    
    # For 2-3 people, try to include them all
    people_to_track = sorted_by_confidence[:min(3, num_people)]
    
    person_boxes = [obj['person_box'] for obj in people_to_track]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * aspect_ratio
    
    # If people fit horizontally, track the group
    if group_width < max_width_for_crop * 1.2:  # Allow 20% overflow (will be handled by crop)
        # Check if group is reasonably sized
        if not would_require_excessive_zoom(group_box, frame_width, frame_height, aspect_ratio, max_zoom):
            return 'TRACK', group_box
    
    # If exactly 2 people are too far apart for single crop, consider stacking
    if num_people == 2:
        person1, person2 = scene_analysis[0], scene_analysis[1]
        box1, box2 = person1['person_box'], person2['person_box']
        
        # Check if boxes overlap significantly (same person detected twice)
        iou = calculate_box_iou(box1, box2)
        if iou > 0.3:  # More than 30% overlap = same person
            best_person = sorted_by_confidence[0]
            print(f"  ⚠️  Duplicate detection (IoU={iou:.2f}), tracking single person")
            return 'TRACK', best_person['person_box']
        
        # Check if people are too close for split-screen
        if are_people_too_close_for_split(person1, person2, frame_width, frame_height, aspect_ratio):
            # People are too close - track the most confident person only
            best_person = sorted_by_confidence[0]
            print(f"  ⚠️  People too close for split-screen, tracking single person")
            return 'TRACK', best_person['person_box']
        
        # Check if people are reasonably sized for stacking
        can_stack = True
        for person in scene_analysis:
            person_box = person['person_box']
            person_height = person_box[3] - person_box[1]
            person_width = person_box[2] - person_box[0]
            
            # More lenient check: person should be at least 2% of frame area
            person_area = person_height * person_width
            frame_area = frame_height * frame_width
            if person_area / frame_area < 0.02:
                can_stack = False
                break
        
        if can_stack:
            # Sort people by horizontal position (left to right) for consistent stacking
            sorted_people = sorted(scene_analysis, key=lambda x: x['person_box'][0])
            return 'STACK', sorted_people
    
    # For 3+ people spread apart, try tracking the two most confident
    if num_people >= 3:
        top_two = sorted_by_confidence[:2]
        two_person_boxes = [obj['person_box'] for obj in top_two]
        two_group_box = get_enclosing_box(two_person_boxes)
        two_group_width = two_group_box[2] - two_group_box[0]
        
        if two_group_width < max_width_for_crop:
            return 'TRACK', two_group_box
        
        # Check if top 2 are actually the same person (significant overlap)
        box1, box2 = top_two[0]['person_box'], top_two[1]['person_box']
        iou = calculate_box_iou(box1, box2)
        if iou > 0.3:
            print(f"  ⚠️  Duplicate detection in top 2 (IoU={iou:.2f}), tracking single person")
            return 'TRACK', sorted_by_confidence[0]['person_box']
        
        # Check if top 2 are too close for split
        if are_people_too_close_for_split(top_two[0], top_two[1], frame_width, frame_height, aspect_ratio):
            print(f"  ⚠️  Top 2 people too close for split-screen, tracking single person")
            return 'TRACK', sorted_by_confidence[0]['person_box']
        
        # Try stacking the top 2
        sorted_two = sorted(top_two, key=lambda x: x['person_box'][0])
        return 'STACK', sorted_two
    
    # Last resort: track the most confident person
    best_person = sorted_by_confidence[0]
    return 'TRACK', best_person['person_box']


def calculate_crop_box(target_box, frame_width, frame_height, aspect_ratio, zoom_factor=1.0):
    """
    Calculate crop box centered on target with optional zoom.
    Ensures returned crop box has EXACT aspect ratio to prevent stretching.
    
    Args:
        target_box: Bounding box [x1, y1, x2, y2]
        frame_width: Frame width
        frame_height: Frame height
        aspect_ratio: Desired aspect ratio (width/height)
        zoom_factor: How much to zoom in (1.0 = full height, 2.0 = tight on person)
    """
    target_center_x = (target_box[0] + target_box[2]) / 2
    target_center_y = (target_box[1] + target_box[3]) / 2
    
    # Calculate ideal crop dimensions based on zoom
    if zoom_factor > 1.0:
        person_height = target_box[3] - target_box[1]
        crop_height = person_height * zoom_factor
        crop_width = crop_height * aspect_ratio
    else:
        # Original behavior: full height crop
        crop_height = frame_height
        crop_width = crop_height * aspect_ratio
    
    # Constrain crop to fit within frame while maintaining aspect ratio
    if crop_width > frame_width:
        crop_width = frame_width
        crop_height = crop_width / aspect_ratio
    if crop_height > frame_height:
        crop_height = frame_height
        crop_width = crop_height * aspect_ratio
    
    # Center on target
    x1 = target_center_x - crop_width / 2
    y1 = target_center_y - crop_height / 2
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    
    # Shift to keep within bounds (maintaining size)
    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    elif x2 > frame_width:
        shift = x2 - frame_width
        x1 -= shift
        x2 -= shift
    
    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    elif y2 > frame_height:
        shift = y2 - frame_height
        y1 -= shift
        y2 -= shift
    
    # Final clamp (should rarely be needed)
    x1 = max(0, min(x1, frame_width - crop_width))
    y1 = max(0, min(y1, frame_height - crop_height))
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    
    # Return as integers
    return [int(x1), int(y1), int(x2), int(y2)]


def calculate_crop_box_centered(center_x, center_y, reference_box, frame_width, frame_height, aspect_ratio, zoom_factor=1.0):
    """
    Calculate crop box centered on a specific point with reference to a bounding box.
    Used for smarter centering in split-screen mode.
    
    Args:
        center_x, center_y: Center point for the crop
        reference_box: Reference bounding box [x1, y1, x2, y2] to calculate dimensions
        frame_width: Frame width
        frame_height: Frame height
        aspect_ratio: Desired aspect ratio (width/height)
        zoom_factor: How much to zoom in (1.0 = full frame, higher = tighter crop around person)
    """
    ref_height = reference_box[3] - reference_box[1]
    ref_width = reference_box[2] - reference_box[0]
    
    # Calculate crop dimensions based on zoom
    # Higher zoom = smaller crop area (tighter on person)
    if zoom_factor > 1.0:
        # Crop should be sized so person fills a good portion of it
        # zoom_factor of 2.0 means person should fill about 50% of crop
        # zoom_factor of 3.0 means person should fill about 66% of crop
        target_person_fill = 1.0 - (1.0 / zoom_factor)  # 2.0 -> 0.5, 3.0 -> 0.66, 4.0 -> 0.75
        target_person_fill = max(0.4, min(target_person_fill, 0.8))  # Clamp to reasonable range
        
        # Calculate crop size based on person filling target percentage
        crop_height = ref_height / target_person_fill
        crop_width = crop_height * aspect_ratio
        
        # If crop width is too narrow for person, adjust
        if crop_width < ref_width / target_person_fill:
            crop_width = ref_width / target_person_fill
            crop_height = crop_width / aspect_ratio
    else:
        # No zoom - use full frame height
        crop_height = frame_height
        crop_width = crop_height * aspect_ratio
    
    # Constrain crop to fit within frame while maintaining aspect ratio
    if crop_width > frame_width:
        crop_width = frame_width
        crop_height = crop_width / aspect_ratio
    if crop_height > frame_height:
        crop_height = frame_height
        crop_width = crop_height * aspect_ratio
    
    # Ensure minimum crop size (at least as big as the person)
    min_crop_height = ref_height * 1.3  # At least 30% padding around person
    min_crop_width = ref_width * 1.3
    if crop_height < min_crop_height:
        crop_height = min(min_crop_height, frame_height)
        crop_width = crop_height * aspect_ratio
    if crop_width < min_crop_width:
        crop_width = min(min_crop_width, frame_width)
        crop_height = crop_width / aspect_ratio
    
    # Center on specified point
    x1 = center_x - crop_width / 2
    y1 = center_y - crop_height / 2
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    
    # Shift to keep within bounds (maintaining size)
    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    elif x2 > frame_width:
        shift = x2 - frame_width
        x1 -= shift
        x2 -= shift
    
    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    elif y2 > frame_height:
        shift = y2 - frame_height
        y1 -= shift
        y2 -= shift
    
    # Final clamp
    x1 = max(0, min(x1, frame_width - crop_width))
    y1 = max(0, min(y1, frame_height - crop_height))
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    
    return [int(x1), int(y1), int(x2), int(y2)]


def resize_cover(image, target_width, target_height):
    """
    Resize image to cover target dimensions (like CSS object-fit: cover).
    Maintains aspect ratio by cropping excess, never stretching.
    
    Args:
        image: Input image (numpy array)
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized and cropped image of exact target dimensions
    """
    img_height, img_width = image.shape[:2]
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height
    
    # Calculate scale to cover entire target area
    if img_aspect > target_aspect:
        # Image is wider - scale by height, crop width
        scale_height = target_height
        scale_width = int(img_width * (target_height / img_height))
    else:
        # Image is taller - scale by width, crop height
        scale_width = target_width
        scale_height = int(img_height * (target_width / img_width))
    
    # Resize to cover dimensions
    resized = cv2.resize(image, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
    
    # Center crop to exact target size
    if scale_width > target_width:
        # Crop width
        x_offset = (scale_width - target_width) // 2
        cropped = resized[:, x_offset:x_offset + target_width]
    elif scale_height > target_height:
        # Crop height
        y_offset = (scale_height - target_height) // 2
        cropped = resized[y_offset:y_offset + target_height, :]
    else:
        # Exact fit (shouldn't happen, but handle it)
        cropped = resized
    
    # Ensure exact dimensions (handle any rounding issues)
    if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    return cropped


def calculate_optimal_zoom(person_box, section_width, section_height, frame_width, frame_height, face_box=None, is_stacking=False):
    """
    Calculate optimal zoom factor to fit person in section.
    Higher zoom = tighter crop around person.
    
    Args:
        person_box: Person bounding box [x1, y1, x2, y2]
        section_width: Target section width
        section_height: Target section height
        frame_width: Original frame width
        frame_height: Original frame height
        face_box: Optional face bounding box for better framing
        is_stacking: If True, use tighter zoom to isolate person in split-screen
        
    Returns:
        Optimal zoom factor (1.0 = full frame, higher = tighter crop)
    """
    person_width = person_box[2] - person_box[0]
    person_height = person_box[3] - person_box[1]
    
    if person_height <= 0 or person_width <= 0:
        return 2.5 if is_stacking else 1.5
    
    if is_stacking:
        # For split-screen: we want a tight crop that isolates this person
        # Person should fill about 50-60% of the crop area
        target_fill = 0.55
        
        # The zoom determines how tight the crop is
        # Higher zoom = person fills more of the crop = smaller crop area
        
        # Base zoom on person size relative to frame
        # If person is small, we need higher zoom to isolate them
        person_frame_ratio = person_height / frame_height
        
        # Smaller person = higher zoom needed
        # Person at 50% of frame -> zoom ~2.0
        # Person at 25% of frame -> zoom ~3.0
        # Person at 10% of frame -> zoom ~4.0
        zoom = target_fill / person_frame_ratio if person_frame_ratio > 0 else 3.0
        
        # Clamp zoom: minimum 2.0 to ensure isolation, max 4.0 to avoid extreme close-ups
        zoom = max(2.0, min(zoom, 4.0))
        
        # If we have a face, check it won't be too large in final output
        if face_box is not None:
            face_height = face_box[3] - face_box[1]
            if face_height > 0:
                # Estimate how much of section the face will fill
                # After crop and resize, face should be max ~30% of section
                crop_height = person_height / target_fill  # Approximate crop height
                face_in_crop_ratio = face_height / crop_height
                # After resize to section, face ratio stays roughly the same
                if face_in_crop_ratio > 0.35:
                    # Face would be too big, reduce zoom
                    adjustment = 0.35 / face_in_crop_ratio
                    zoom = zoom * adjustment
                    zoom = max(2.0, zoom)
    else:
        # Regular tracking mode - gentler zoom
        target_fill = 0.5
        person_frame_ratio = person_height / frame_height
        zoom = target_fill / person_frame_ratio if person_frame_ratio > 0 else 1.5
        
        # Conservative range for tracking
        zoom = max(1.2, min(zoom, 2.5))
    
    return zoom


def get_smart_center_point(person_box, face_box=None):
    """
    Calculate a smart center point for cropping.
    Uses face position to bias towards upper body while keeping person in frame.
    
    Args:
        person_box: Person bounding box [x1, y1, x2, y2]
        face_box: Optional face bounding box [x1, y1, x2, y2]
        
    Returns:
        (center_x, center_y) tuple
    """
    person_center_x = (person_box[0] + person_box[2]) / 2
    person_center_y = (person_box[1] + person_box[3]) / 2
    
    if face_box is not None:
        face_center_x = (face_box[0] + face_box[2]) / 2
        face_center_y = (face_box[1] + face_box[3]) / 2
        
        # Blend between person center and a point above face center
        # This keeps the face in upper third while showing more body
        # Horizontal: use face center for better face positioning
        # Vertical: bias towards showing head + upper body
        center_x = face_center_x
        
        # Place center below face to show upper body
        # Face should be in upper 30-40% of the crop
        face_height = face_box[3] - face_box[1]
        center_y = face_center_y + face_height * 1.5  # Shift down to show more body
        
        # But don't go beyond person center (would cut off head)
        center_y = min(center_y, person_center_y)
        
        return (center_x, center_y)
    
    # No face detected - use upper portion of person box
    # Assume head is at top, so shift center up slightly
    person_height = person_box[3] - person_box[1]
    center_y = person_box[1] + person_height * 0.35  # Upper third
    
    return (person_center_x, center_y)


def create_stacked_frame(frame, people_data, output_width, output_height, aspect_ratio):
    """
    Create a frame with 2 people stacked based on aspect ratio.
    Improved centering and zoom to avoid excessive face cropping.
    
    Args:
        frame: Original video frame
        people_data: List of exactly 2 person detection data (sorted by position)
        output_width: Target output width
        output_height: Target output height
        aspect_ratio: Target aspect ratio (width/height)
        
    Returns:
        Stacked frame with 2 people arranged optimally for aspect ratio
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Create output frame
    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Determine stacking strategy based on aspect ratio
    if aspect_ratio < 0.8:  # Vertical (9:16 = 0.5625, portrait)
        # Stack vertically (top to bottom)
        section_height = output_height // 2
        
        for i, person in enumerate(people_data):
            person_box = person['person_box']
            face_box = person.get('face_box')
            
            # Calculate section bounds
            y_start = i * section_height
            y_end = y_start + section_height
            if i == 1 and y_end < output_height:  # Last person gets any remaining pixels
                y_end = output_height
            actual_section_height = y_end - y_start
            actual_section_aspect = output_width / actual_section_height
            
            # Calculate optimal zoom with is_stacking=True for tighter isolation
            zoom = calculate_optimal_zoom(person_box, output_width, actual_section_height, 
                                         frame_width, frame_height, face_box=face_box, is_stacking=True)
            
            # Get smart center point that considers face position
            center_x, center_y = get_smart_center_point(person_box, face_box)
            
            # Calculate crop box centered on smart point
            crop_box = calculate_crop_box_centered(
                center_x, center_y, person_box,
                frame_width, frame_height, actual_section_aspect, zoom_factor=zoom
            )
            person_crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            # Resize using "cover" method (no stretching)
            if person_crop.size > 0:
                person_resized = resize_cover(person_crop, output_width, actual_section_height)
                output_frame[y_start:y_end, :] = person_resized
    
    elif aspect_ratio > 1.2:  # Horizontal (16:9 = 1.778, landscape)
        # Stack horizontally (side by side)
        section_width = output_width // 2
        
        for i, person in enumerate(people_data):
            person_box = person['person_box']
            face_box = person.get('face_box')
            
            # Calculate section bounds
            x_start = i * section_width
            x_end = x_start + section_width
            if i == 1 and x_end < output_width:  # Last person gets any remaining pixels
                x_end = output_width
            actual_section_width = x_end - x_start
            
            # Each section maintains aspect ratio
            section_aspect = actual_section_width / output_height
            
            # Calculate optimal zoom with is_stacking=True
            zoom = calculate_optimal_zoom(person_box, actual_section_width, output_height, 
                                         frame_width, frame_height, face_box=face_box, is_stacking=True)
            
            # Get smart center point
            center_x, center_y = get_smart_center_point(person_box, face_box)
            
            # Calculate crop box centered on smart point
            crop_box = calculate_crop_box_centered(
                center_x, center_y, person_box,
                frame_width, frame_height, section_aspect, zoom_factor=zoom
            )
            person_crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            # Resize using "cover" method (no stretching)
            if person_crop.size > 0:
                person_resized = resize_cover(person_crop, actual_section_width, output_height)
                output_frame[:, x_start:x_end] = person_resized
    
    else:  # Square-ish (1:1 = 1.0)
        # Side by side for 2 people
        section_width = output_width // 2
        section_aspect = section_width / output_height
        
        for i, person in enumerate(people_data):
            person_box = person['person_box']
            face_box = person.get('face_box')
            
            # Calculate optimal zoom with is_stacking=True
            zoom = calculate_optimal_zoom(person_box, section_width, output_height, 
                                         frame_width, frame_height, face_box=face_box, is_stacking=True)
            
            # Get smart center point
            center_x, center_y = get_smart_center_point(person_box, face_box)
            
            # Calculate crop box centered on smart point
            crop_box = calculate_crop_box_centered(
                center_x, center_y, person_box,
                frame_width, frame_height, section_aspect, zoom_factor=zoom
            )
            person_crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            # Resize using "cover" method (no stretching)
            if person_crop.size > 0:
                person_resized = resize_cover(person_crop, section_width, output_height)
                x_start = i * section_width
                output_frame[:, x_start:x_start + section_width] = person_resized
    
    return output_frame


class SmoothTracker:
    """
    Smooth tracking system for OpusClip-like camera movement.
    Uses dead zone + adaptive smoothing to avoid jitter from small movements.
    Only follows when subject moves significantly.
    """
    
    def __init__(self, smoothing=0.08, dead_zone_ratio=0.03):
        """
        Args:
            smoothing: Base smoothing factor (0.05-0.15 recommended)
            dead_zone_ratio: Fraction of frame width/height to ignore as dead zone (default 3%)
        """
        self.base_smoothing = smoothing
        self.dead_zone_ratio = dead_zone_ratio
        self.current_x = None
        self.current_y = None
        self.target_x = None  # Smoothed target (not actual position)
        self.target_y = None
        self.frame_width = 1920  # Will be updated on first use
        self.frame_height = 1080
    
    def set_frame_size(self, width, height):
        """Set frame dimensions for dead zone calculation."""
        self.frame_width = width
        self.frame_height = height
    
    def update(self, target_x, target_y):
        """Update tracker with new target position, returns smoothed position."""
        if self.current_x is None:
            self.current_x = target_x
            self.current_y = target_y
            self.target_x = target_x
            self.target_y = target_y
            return self.current_x, self.current_y
        
        # Calculate dead zone thresholds (% of frame size)
        dead_zone_x = self.frame_width * self.dead_zone_ratio
        dead_zone_y = self.frame_height * self.dead_zone_ratio
        
        # Calculate distance from current smoothed target to new target
        dx = target_x - self.target_x
        dy = target_y - self.target_y
        
        # Only update target if movement exceeds dead zone
        # This prevents camera from chasing every micro-movement
        if abs(dx) > dead_zone_x:
            # Move target, but keep it slightly inside the dead zone
            # This creates a "lazy follow" effect
            if dx > 0:
                self.target_x = target_x - dead_zone_x * 0.5
            else:
                self.target_x = target_x + dead_zone_x * 0.5
        
        if abs(dy) > dead_zone_y:
            if dy > 0:
                self.target_y = target_y - dead_zone_y * 0.5
            else:
                self.target_y = target_y + dead_zone_y * 0.5
        
        # Calculate distance for adaptive smoothing
        dist_to_target = ((self.target_x - self.current_x) ** 2 + 
                          (self.target_y - self.current_y) ** 2) ** 0.5
        
        # Adaptive smoothing: slower for small movements, faster for large
        # This prevents overshooting on small adjustments
        frame_diagonal = (self.frame_width ** 2 + self.frame_height ** 2) ** 0.5
        movement_ratio = dist_to_target / frame_diagonal
        
        # Scale smoothing: very slow for tiny movements, up to base for large movements
        # min 0.02 (very slow), max = base_smoothing
        adaptive_smoothing = self.base_smoothing * min(1.0, movement_ratio * 10)
        adaptive_smoothing = max(0.02, adaptive_smoothing)
        
        # Apply smoothing to move toward target
        self.current_x += adaptive_smoothing * (self.target_x - self.current_x)
        self.current_y += adaptive_smoothing * (self.target_y - self.current_y)
        
        return self.current_x, self.current_y
    
    def reset(self):
        """Reset tracker state."""
        self.current_x = None
        self.current_y = None
        self.target_x = None
        self.target_y = None
    
    def snap_to(self, x, y):
        """Instantly snap to position (for scene changes)."""
        self.current_x = x
        self.current_y = y
        self.target_x = x
        self.target_y = y


def track_subjects_in_frame(frame, model, previous_boxes=None, confidence_threshold=0.25):
    """
    Track any subjects (people, objects) in a frame using YOLO.
    Returns the best target to follow.
    
    Args:
        frame: Current video frame
        model: YOLO model
        previous_boxes: Previous frame's detections for continuity
        confidence_threshold: Minimum detection confidence
        
    Returns:
        Best target box [x1, y1, x2, y2] or None
    """
    results = model([frame], verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if confidence < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            
            # Prioritize people (class 0), but track other objects too
            priority = 10 if cls == 0 else 1
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': confidence,
                'class': cls,
                'priority': priority,
                'area': (x2 - x1) * (y2 - y1)
            })
    
    if not detections:
        return None
    
    # If we have previous boxes, try to find the same subject
    if previous_boxes:
        best_match = None
        best_iou = 0.2
        
        for det in detections:
            for prev in previous_boxes:
                iou = calculate_iou(det['box'], prev['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
        
        if best_match:
            return best_match['box']
    
    # Sort by priority (people first), then by area (larger = more important)
    detections.sort(key=lambda x: (x['priority'], x['area']), reverse=True)
    
    return detections[0]['box']


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)


def calculate_smooth_crop(target_center_x, target_center_y, frame_width, frame_height, aspect_ratio):
    """
    Calculate crop box centered on a smooth target position.
    """
    crop_height = frame_height
    crop_width = crop_height * aspect_ratio
    
    # Constrain to frame
    if crop_width > frame_width:
        crop_width = frame_width
        crop_height = crop_width / aspect_ratio
    
    # Center on target
    x1 = target_center_x - crop_width / 2
    y1 = target_center_y - crop_height / 2
    
    # Keep within bounds
    x1 = max(0, min(x1, frame_width - crop_width))
    y1 = max(0, min(y1, frame_height - crop_height))
    
    return [int(x1), int(y1), int(x1 + crop_width), int(y1 + crop_height)]


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def get_video_codec(video_path):
    """Detect the video codec using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return None


# Codecs that OpenCV can't decode properly and need transcoding
PROBLEMATIC_CODECS = {'av1', 'vp9', 'vp8', 'hevc', 'h265'}


def transcode_to_h264(input_path, output_path=None):
    """
    Transcode video to H.264 for OpenCV compatibility.
    Returns the path to the transcoded file.
    """
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '_h264.mp4'
    
    print(f"🔄 Transcoding to H.264 for processing compatibility...")
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Transcoding complete")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Transcoding failed: {e.stderr}")
        return None


def process_video(input_video, final_output_video, model, face_cascade, aspect_ratio=9/16, analysis_scale=1.0, use_gpu=False, confidence_threshold=0.3, use_dnn_face=True, num_sample_frames=3, detect_speaker=False, fallback_strategy='saliency', tracking_mode='smooth', tracking_smoothness=0.08, verbose=False, face_model=None):
    """
    Main video processing function that converts horizontal video to vertical format.
    OpusClip-like quality with smooth tracking and real-time subject following.
    
    Args:
        input_video: Path to the input video file
        final_output_video: Path to the output video file
        model: YOLO model instance
        face_cascade: Haar Cascade face detector instance (fallback)
        aspect_ratio: Target aspect ratio (width/height), default is 9/16 for vertical video
        analysis_scale: Scale factor for scene analysis (e.g., 0.5 = half resolution for faster processing)
        use_gpu: Whether GPU is available for hardware acceleration
        confidence_threshold: Minimum confidence (0-1) for person detection. Default: 0.3 (lowered for better recall)
        use_dnn_face: Whether to use enhanced face detection (YOLO face model if available, or estimation from person boxes). Default: True
        num_sample_frames: Number of frames to sample per scene for detection (default: 3). More = better accuracy, slower.
        detect_speaker: Use fast MediaPipe-based speaker detection to focus on who's talking. Default: False
        fallback_strategy: Strategy when no people detected ('saliency', 'center', 'letterbox'). Default: 'saliency'
        tracking_mode: 'smooth' (real-time tracking with smoothing), 'static' (per-scene), 'fast' (real-time, less smooth). Default: 'smooth'
        tracking_smoothness: Camera smoothness (0.05=very smooth, 0.15=responsive). Default: 0.08
        verbose: Show detailed debug info for each scene. Default: False
        face_model: Pre-loaded YOLO face model (optional). If None and use_dnn_face=True, will try to load one.
    """
    script_start_time = time.time()
    
    print(f"🎬 Autocrop v{AUTOCROP_VERSION}")
    print(f"=" * 40)
    
    # Check if video is already close to the target aspect ratio
    original_width, original_height = get_video_resolution(input_video)
    current_aspect_ratio = original_width / original_height
    aspect_ratio_diff = abs(current_aspect_ratio - aspect_ratio) / aspect_ratio
    
    # If within 20% of target aspect ratio, just copy the original video
    if aspect_ratio_diff < 0.20:
        print(f"✅ Video aspect ratio ({current_aspect_ratio:.3f}) is already close to target ({aspect_ratio:.3f})")
        print("   Copying original video without processing...")
        shutil.copy2(input_video, final_output_video)
        print(f"✅ Done! Output saved to: {final_output_video}")
        return
    
    # Check if video codec is problematic for OpenCV (AV1, VP9, etc.)
    transcoded_input = None
    video_codec = get_video_codec(input_video)
    if video_codec and video_codec.lower() in PROBLEMATIC_CODECS:
        print(f"⚠️  Detected {video_codec.upper()} codec - OpenCV cannot decode this properly")
        base_name = os.path.splitext(final_output_video)[0]
        transcoded_input = f"{base_name}_transcoded_input.mp4"
        transcoded_path = transcode_to_h264(input_video, transcoded_input)
        if transcoded_path:
            input_video = transcoded_path
            # Update resolution after transcoding
            original_width, original_height = get_video_resolution(input_video)
        else:
            print("❌ Failed to transcode video. Processing may fail.")
    
    # Load face detector (YOLO face model if available, otherwise uses estimation from person boxes)
    dnn_face_detector = None
    if use_dnn_face:
        if face_model is not None:
            # Use pre-loaded face model (faster, avoids reloading)
            dnn_face_detector = face_model
            print("✅ Using pre-loaded YOLO face model")
        else:
            dnn_face_detector = load_yolo_face_detector()
    
    # Initialize fast speaker detection if enabled
    if detect_speaker:
        print("🎤 Initializing fast speaker detection (MediaPipe)...")
        init_mediapipe()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.m4a"  # Use .m4a to support any audio codec
    
    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output):
        os.remove(temp_video_output)
    if os.path.exists(temp_audio_output):
        os.remove(temp_audio_output)
    if os.path.exists(final_output_video):
        os.remove(final_output_video)

    print("🎬 Step 1: Detecting scenes...")
    step_start_time = time.time()
    scenes, fps = detect_scenes(input_video)
    step_end_time = time.time()
    
    if not scenes:
        print("❌ No scenes were detected. Aborting.")
        raise ValueError("No scenes detected in video")
    
    print(f"✅ Found {len(scenes)} scenes in {step_end_time - step_start_time:.2f}s. Here is the breakdown:")
    for i, (start, end) in enumerate(scenes):
        print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")

    speaker_info = " + speaker detection" if detect_speaker else ""
    fallback_info = f" + {fallback_strategy} fallback" if fallback_strategy != 'letterbox' else ""
    print(f"\n🧠 Step 2: Analyzing scene content (multi-frame sampling: {num_sample_frames} frames/scene{speaker_info}{fallback_info})...")
    step_start_time = time.time()
    # original_width, original_height already obtained above
    
    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * aspect_ratio)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    scenes_analysis = []
    speakers_detected = 0
    conversations_detected = 0
    
    for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing Scenes")):
        # Analyze scene content
        analysis = analyze_scene_content(input_video, start_time, end_time, model, face_cascade, analysis_scale, 
                                        dnn_face_detector=dnn_face_detector, confidence_threshold=confidence_threshold,
                                        num_sample_frames=num_sample_frames)
        
        # Fast speaker detection if enabled and multiple people
        active_speaker_idx = None
        is_conversation = False
        
        if detect_speaker and len(analysis) > 1 and MEDIAPIPE_AVAILABLE:
            active_speaker_idx, is_conversation, _ = detect_speaker_fast(
                input_video,
                start_time.get_frames(),
                end_time.get_frames(),
                analysis,
                fps
            )
            if active_speaker_idx is not None:
                speakers_detected += 1
            if is_conversation:
                conversations_detected += 1
        
        # Decide cropping strategy
        # Conversations use letterbox to show both people
        if is_conversation and len(analysis) >= 2:
            # Conversation detected - use letterbox to show everyone
            print(f"  💬 Conversation: using letterbox to show both speakers")
            strategy = 'LETTERBOX'
            target_box = None
        elif active_speaker_idx is not None:
            # Clear speaker detected - focus on them
            strategy, target_box = decide_cropping_strategy(
                analysis, original_height, original_width, aspect_ratio,
                active_speaker_idx=active_speaker_idx
            )
        else:
            # Normal strategy
            strategy, target_box = decide_cropping_strategy(
                analysis, original_height, original_width, aspect_ratio
            )
        
        # If no people detected (LETTERBOX but NOT conversation), try fallback strategy
        # Don't override conversation letterbox - that's intentional
        if strategy == 'LETTERBOX' and not is_conversation and len(analysis) == 0:
            if fallback_strategy != 'letterbox':
                fallback_box = compute_scene_fallback(
                    input_video, start_time, end_time, aspect_ratio, fallback_strategy
                )
                if fallback_box:
                    strategy = 'TRACK'
                    target_box = fallback_box
            # If saliency fails or fallback_strategy is letterbox, keep LETTERBOX
        
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box,
            'active_speaker': active_speaker_idx,
            'is_conversation': is_conversation
        })
    
    step_end_time = time.time()
    speaker_msg = ""
    if detect_speaker and (speakers_detected > 0 or conversations_detected > 0):
        parts = []
        if speakers_detected > 0:
            parts.append(f"{speakers_detected} speaker(s) detected")
        if conversations_detected > 0:
            parts.append(f"{conversations_detected} conversation(s)")
        speaker_msg = f" ({', '.join(parts)})"
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.{speaker_msg}")

    print("\n📋 Step 3: Generated Processing Plan")
    for i, scene_data in enumerate(scenes_analysis):
        num_people = len(scene_data['analysis'])
        strategy = scene_data['strategy']
        start_time = scenes[i][0].get_timecode()
        end_time = scenes[i][1].get_timecode()
        active_speaker = scene_data.get('active_speaker')
        is_conversation = scene_data.get('is_conversation', False)
        
        # Calculate average confidence for this scene
        if num_people > 0:
            avg_person_conf = sum(obj['confidence'] for obj in scene_data['analysis']) / num_people
            face_confs = [obj.get('face_confidence', 0) for obj in scene_data['analysis'] if obj.get('face_confidence', 0) > 0]
            
            # Check multi-frame detection consistency
            frames_info = ""
            if 'frames_detected' in scene_data['analysis'][0]:
                consistent = sum(1 for obj in scene_data['analysis'] if obj.get('frames_detected', 1) >= obj.get('total_frames', 1) * 0.5)
                total_frames = scene_data['analysis'][0].get('total_frames', 1)
                if total_frames > 1:
                    frames_info = f" ({consistent}/{num_people} stable)"
            
            # Speaker info
            speaker_note = ""
            if is_conversation:
                speaker_note = " 💬CONV"
            elif active_speaker is not None:
                speaker_note = f" 🎤P{active_speaker + 1}"
            
            if face_confs:
                avg_face_conf = sum(face_confs) / len(face_confs)
                faces_detected = len(face_confs)
                print(f"  - Scene {i+1} ({start_time} -> {end_time}): {num_people} person(s) [conf: {avg_person_conf:.2f}]{frames_info}, {faces_detected} face(s) [conf: {avg_face_conf:.2f}]{speaker_note}. Strategy: {strategy}")
            else:
                print(f"  - Scene {i+1} ({start_time} -> {end_time}): {num_people} person(s) [conf: {avg_person_conf:.2f}]{frames_info}, 0 faces{speaker_note}. Strategy: {strategy}")
        else:
            fallback_note = " (saliency)" if strategy == 'TRACK' else " ⚠️ LETTERBOX"
            print(f"  - Scene {i+1} ({start_time} -> {end_time}): 0 person(s). Strategy: {strategy}{fallback_note}")
        
        # Verbose mode: show why each scene got its strategy
        if verbose and num_people > 0:
            for j, person in enumerate(scene_data['analysis']):
                box = person['person_box']
                box_size = f"{box[2]-box[0]}x{box[3]-box[1]}"
                print(f"      Person {j+1}: box={box_size}, conf={person['confidence']:.2f}")

    print("\n✂️ Step 4: Processing video frames...")
    step_start_time = time.time()
    
    # Build FFmpeg command with hardware acceleration if available
    if use_gpu:
        print("  🚀 Using NVIDIA GPU hardware encoding (h264_nvenc)...")
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
            '-r', str(fps), '-i', '-',
            '-c:v', 'h264_nvenc',          # NVIDIA hardware encoder
            '-preset', 'p4',                # P4 preset (balanced quality/speed)
            '-tune', 'hq',                  # High quality tuning
            '-rc', 'vbr',                   # Variable bitrate
            '-rc-lookahead', '20',          # Lookahead for better quality
            '-spatial_aq', '1',             # Spatial adaptive quantization
            '-temporal_aq', '1',            # Temporal adaptive quantization
            '-cq', '23',                    # Quality level (like CRF)
            '-b:v', '0',                    # Use quality target instead of bitrate
            '-g', str(int(fps * 2)),        # Keyframe every 2 seconds
            '-bf', '3',                     # B-frames for better compression
            '-movflags', '+faststart',      # Fast start for web playback
            '-an', temp_video_output
        ]
    else:
        print("  💻 Using CPU software encoding (libx264)...")
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
            '-r', str(fps), '-i', '-',
            '-c:v', 'libx264',              # CPU software encoder
            '-preset', 'ultrafast',         # Fastest CPU preset
            '-crf', '23',                   # Quality level
            '-an', temp_video_output
        ]

    # Large buffer to prevent pipe deadlocks
    try:
        ffmpeg_process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # 100MB buffer
        )
    except FileNotFoundError:
        if use_gpu:
            print("  ⚠️  h264_nvenc not available, falling back to CPU encoding...")
            use_gpu = False  # Disable GPU for this session
            command = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
                '-r', str(fps), '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast', '-crf', '23', '-an', temp_video_output
            ]
            ffmpeg_process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
    
    # Thread to consume stderr and prevent blocking
    stderr_lines = []
    def read_stderr():
        try:
            for line in iter(ffmpeg_process.stderr.readline, b''):
                stderr_lines.append(line.decode('utf-8', errors='ignore'))
        except:
            pass
    
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set buffer size to prevent memory buildup
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    frame_number = 0
    current_scene_index = 0
    
    # Initialize smooth tracker for OpusClip-like camera movement
    use_realtime = tracking_mode in ['smooth', 'fast']
    smoothness = tracking_smoothness if tracking_mode == 'smooth' else 0.2
    tracker = SmoothTracker(smoothing=smoothness)
    tracker.set_frame_size(original_width, original_height)
    
    # For real-time tracking
    previous_detections = None
    # Track interval: detect every N frames
    # With GPU (L40S), we can track more frequently for better responsiveness
    # Without GPU, track less often to save CPU
    if use_gpu:
        track_interval = 2 if tracking_mode == 'smooth' else 4  # GPU: more frequent tracking
    else:
        track_interval = 5 if tracking_mode == 'smooth' else 8  # CPU: less frequent
    last_detected_box = None
    
    # Pre-allocate letterbox frame to avoid repeated allocations
    letterbox_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    
    mode_desc = {
        'smooth': '🎬 OpusClip-like smooth tracking',
        'fast': '⚡ Fast tracking',
        'static': '📌 Static per-scene'
    }
    gpu_info = f" (GPU: every {track_interval} frames)" if use_gpu else f" (CPU: every {track_interval} frames)"
    print(f"  {mode_desc.get(tracking_mode, '📌 Static')}{gpu_info if use_realtime else ''}...")
    
    with tqdm(total=total_frames, desc="Processing", smoothing=0.1) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect scene change
            prev_scene = current_scene_index
            if current_scene_index < len(scenes_analysis) - 1 and \
               frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                current_scene_index += 1
                # Reset tracking on scene change
                previous_detections = None
                last_detected_box = None

            scene_data = scenes_analysis[current_scene_index]
            strategy = scene_data['strategy']
            target_box = scene_data['target_box']

            if strategy == 'TRACK':
                # Real-time tracking: detect subject position every few frames
                if use_realtime and frame_number % track_interval == 0:
                    detected = track_subjects_in_frame(
                        frame, model, 
                        [{'box': last_detected_box}] if last_detected_box else None,
                        confidence_threshold * 0.8
                    )
                    if detected:
                        last_detected_box = detected
                        previous_detections = [{'box': detected}]
                
                # Use real-time detection or fallback to scene analysis
                current_target = last_detected_box if last_detected_box else target_box
                
                # Calculate target center
                target_cx = (current_target[0] + current_target[2]) / 2
                target_cy = (current_target[1] + current_target[3]) / 2
                
                # Apply smooth tracking
                if tracking_mode == 'static':
                    smooth_cx, smooth_cy = target_cx, target_cy
                else:
                    # On scene change, snap to new position then smooth
                    if prev_scene != current_scene_index:
                        tracker.snap_to(target_cx, target_cy)
                    smooth_cx, smooth_cy = tracker.update(target_cx, target_cy)
                
                # Calculate smooth crop
                crop_box = calculate_smooth_crop(
                    smooth_cx, smooth_cy, 
                    original_width, original_height, aspect_ratio
                )
                
                # Extract and resize
                x1, y1, x2, y2 = crop_box
                processed_frame = frame[y1:y2, x1:x2]
                if processed_frame.size > 0:
                    output_frame = resize_cover(processed_frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                else:
                    output_frame = letterbox_frame.copy()
                    
            elif strategy == 'STACK':
                # Stack multiple people
                people_data = target_box
                output_frame = create_stacked_frame(frame, people_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, aspect_ratio)
                tracker.reset()
                last_detected_box = None
                
            else:  # LETTERBOX
                scale_factor = OUTPUT_WIDTH / original_width
                scaled_height = int(original_height * scale_factor)
                scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height), interpolation=cv2.INTER_LINEAR)
                
                letterbox_frame.fill(0)
                y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
                letterbox_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
                output_frame = letterbox_frame
                tracker.reset()
                last_detected_box = None
            
            # Write to FFmpeg pipe
            try:
                ffmpeg_process.stdin.write(output_frame.tobytes())
            except BrokenPipeError:
                print("\n⚠️  FFmpeg pipe broken, stopping...")
                break
                
            frame_number += 1
            pbar.update(1)
            
            # Cleanup every 1000 frames
            if frame_number % 1000 == 0:
                del frame
                if strategy == 'TRACK':
                    del processed_frame
    
    ffmpeg_process.stdin.close()
    ffmpeg_process.stdout.close()
    
    # Wait for FFmpeg to finish with timeout
    try:
        ffmpeg_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("\n⚠️  FFmpeg process timeout, terminating...")
        ffmpeg_process.kill()
        ffmpeg_process.wait()
    
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n❌ FFmpeg frame processing failed.")
        if stderr_lines:
            print("FFmpeg errors:")
            for line in stderr_lines[-20:]:  # Show last 20 lines
                print("  ", line.strip())
        raise RuntimeError("FFmpeg frame processing failed")
    step_end_time = time.time()
    print(f"✅ Video processing complete in {step_end_time - step_start_time:.2f}s.")

    print("\n🔊 Step 5: Extracting original audio...")
    step_start_time = time.time()
    
    # First, try to copy audio stream as-is
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    
    audio_extracted = False
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        audio_extracted = True
        print(f"✅ Audio extracted in {time.time() - step_start_time:.2f}s.")
    except subprocess.CalledProcessError as e:
        # If copy fails (e.g., Opus audio), transcode to AAC
        print("  ⚠️  Direct audio copy failed, transcoding to AAC...")
        temp_audio_output = f"{base_name}_temp_audio.aac"  # Switch to AAC extension
        audio_transcode_command = [
            'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'aac', '-b:a', '192k', temp_audio_output
        ]
        try:
            subprocess.run(audio_transcode_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            audio_extracted = True
            print(f"  ✅ Audio transcoded in {time.time() - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e2:
            print("  ⚠️  No audio stream found or audio extraction failed, continuing without audio...")
            audio_extracted = False

    if audio_extracted:
        print("\n✨ Step 6: Merging video and audio...")
        step_start_time = time.time()
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', final_output_video
        ]
        try:
            subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Final video merged in {step_end_time - step_start_time:.2f}s.")
            
            # Clean up temp files after successful merge
            if os.path.exists(temp_video_output):
                os.remove(temp_video_output)
            if os.path.exists(temp_audio_output):
                os.remove(temp_audio_output)
        except subprocess.CalledProcessError as e:
            print("\n❌ Final merge failed.")
            print("Stderr:", e.stderr.decode())
            raise RuntimeError("Final merge failed")
    else:
        # No audio to merge, just rename the video file
        print("\n✨ Step 6: Finalizing video (no audio)...")
        os.rename(temp_video_output, final_output_video)
        print("✅ Video finalized.")

    script_end_time = time.time()
    print(f"\n🎉 All done! Final video saved to {final_output_video}")
    print(f"⏱️  Total execution time: {script_end_time - script_start_time:.2f} seconds.")
    
    # Clean up transcoded input file if it was created
    if transcoded_input and os.path.exists(transcoded_input):
        os.remove(transcoded_input)
