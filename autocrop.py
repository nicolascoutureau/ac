import time
import cv2
import scenedetect
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os
import numpy as np
from tqdm import tqdm
import threading
import queue

def analyze_scene_content(video_path, scene_start_time, scene_end_time, model, face_cascade, analysis_scale=1.0, dnn_face_detector=None, confidence_threshold=0.5):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    
    Args:
        confidence_threshold: Minimum confidence (0-1) for YOLO person detection. Default: 0.5
    Uses fallback detection if primary method doesn't find anything.
    
    Args:
        analysis_scale: Scale factor for analysis (e.g., 0.5 = half resolution). Lower = faster but less accurate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

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

    # Primary detection: YOLO for people (using provided confidence threshold)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:  # Person class
                confidence = float(box.conf[0])
                
                # Filter by confidence threshold
                if confidence < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                
                # Scale bounding box back to original resolution if needed
                x1 = int(x1 * scale_back_x)
                y1 = int(y1 * scale_back_y)
                x2 = int(x2 * scale_back_x)
                y2 = int(y2 * scale_back_y)
                
                person_box = [x1, y1, x2, y2]
                
                # Detect face within person box on original resolution frame
                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                face_box = None
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({
                    'person_box': person_box, 
                    'face_box': face_box,
                    'confidence': confidence
                })
    
    # Fallback detection: Use Haar Cascade on full frame if YOLO found no people
    if len(detected_objects) == 0:
        print("  ⚠️  YOLO detected no people, trying Haar Cascade fallback...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            face_box = [x, y, x + w, y + h]
            # Create a person box slightly larger than face box
            padding = int(max(w, h) * 0.5)
            person_box = [
                max(0, x - padding),
                max(0, y - padding),
                min(frame_width, x + w + padding),
                min(frame_height, y + h + padding)
            ]
            detected_objects.append({
                'person_box': person_box, 
                'face_box': face_box,
                'confidence': 0.3  # Lower confidence for fallback detection
            })
            break  # Just use the first face found
    
    cap.release()
    return detected_objects


def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
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


def would_require_excessive_zoom(target_box, frame_width, frame_height, aspect_ratio, max_zoom=3.0):
    """
    Check if tracking this target would require excessive zoom.
    
    Args:
        target_box: Bounding box [x1, y1, x2, y2]
        frame_width: Frame width
        frame_height: Frame height
        aspect_ratio: Target aspect ratio
        max_zoom: Maximum acceptable zoom factor
        
    Returns:
        True if zoom would exceed max_zoom
    """
    target_height = target_box[3] - target_box[1]
    target_width = target_box[2] - target_box[0]
    
    # Calculate what crop dimensions would be needed
    output_width = int(frame_height * aspect_ratio)
    output_height = frame_height
    
    # Calculate zoom needed (similar logic to calculate_optimal_zoom)
    target_fill = 0.7
    section_aspect = output_width / output_height
    target_aspect = target_width / target_height
    
    if target_aspect > section_aspect:
        desired_crop_width = target_width / target_fill
        desired_crop_height = desired_crop_width / section_aspect
        required_zoom = desired_crop_height / target_height
    else:
        desired_crop_height = target_height / target_fill
        required_zoom = desired_crop_height / target_height
    
    return required_zoom > max_zoom


def decide_cropping_strategy(scene_analysis, frame_height, frame_width, aspect_ratio, max_zoom=3.0):
    """
    Decide the cropping strategy based on scene content.
    Falls back to LETTERBOX if tracking would require excessive zoom.
    
    Args:
        max_zoom: Maximum acceptable zoom factor before falling back to letterbox (default: 3.0)
    """
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        
        # Check if zoom would be excessive for single person
        if would_require_excessive_zoom(target_box, frame_width, frame_height, aspect_ratio, max_zoom):
            print(f"  ⚠️  Person too small (zoom > {max_zoom}x), using LETTERBOX")
            return 'LETTERBOX', None
        
        return 'TRACK', target_box
    
    # Multiple people detected
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * aspect_ratio
    
    # If people fit horizontally, track the group
    if group_width < max_width_for_crop:
        # Check if zoom would be excessive for group
        if would_require_excessive_zoom(group_box, frame_width, frame_height, aspect_ratio, max_zoom):
            print(f"  ⚠️  Group too small (zoom > {max_zoom}x), using LETTERBOX")
            return 'LETTERBOX', None
        
        return 'TRACK', group_box
    
    # If exactly 2 people are too far apart, check if stacking would need excessive zoom
    if num_people == 2:
        # Check if individual people are too small for stacking
        # For stacking, each person gets half the output space
        output_width = int(frame_height * aspect_ratio)
        output_height = frame_height
        
        # Check zoom needed for each person in their section
        section_height = output_height // 2 if aspect_ratio < 0.8 else output_height
        section_width = output_width // 2 if aspect_ratio > 1.2 else output_width
        
        for person in scene_analysis:
            person_box = person['face_box'] or person['person_box']
            person_height = person_box[3] - person_box[1]
            person_width = person_box[2] - person_box[0]
            
            # Rough estimate: if person is less than 1/6 of frame, they'll need >3x zoom
            if person_height < frame_height / 6 or person_width < frame_width / 6:
                print(f"  ⚠️  People too small for stacking (zoom > {max_zoom}x), using LETTERBOX")
                return 'LETTERBOX', None
        
        # Sort people by horizontal position (left to right)
        sorted_people = sorted(scene_analysis, key=lambda x: x['person_box'][0])
        return 'STACK', sorted_people
    
    # Too many people or they can't be stacked nicely, use letterbox
    return 'LETTERBOX', None


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


def calculate_optimal_zoom(person_box, section_width, section_height, frame_width, frame_height):
    """
    Calculate optimal zoom factor to fit person in section without stretching.
    
    Args:
        person_box: Person bounding box [x1, y1, x2, y2]
        section_width: Target section width
        section_height: Target section height
        frame_width: Original frame width
        frame_height: Original frame height
        
    Returns:
        Optimal zoom factor (1.0 = full frame, higher = tighter)
    """
    person_width = person_box[2] - person_box[0]
    person_height = person_box[3] - person_box[1]
    
    # Calculate how much of the section we want the person to fill (60-80%)
    target_fill = 0.7
    
    # Calculate section aspect ratio
    section_aspect = section_width / section_height
    
    # Determine limiting dimension
    # If person is wider relative to section aspect, width is limiting
    person_aspect = person_width / person_height
    
    if person_aspect > section_aspect:
        # Person is wide - width is limiting dimension
        # Calculate zoom based on how much of section width person should fill
        desired_crop_width = person_width / target_fill
        desired_crop_height = desired_crop_width / section_aspect
        zoom = desired_crop_height / person_height
    else:
        # Person is tall - height is limiting dimension
        # Calculate zoom based on how much of section height person should fill
        desired_crop_height = person_height / target_fill
        zoom = desired_crop_height / person_height
    
    # Clamp zoom to reasonable range
    zoom = max(1.3, min(zoom, 2.5))  # Between 1.3x and 2.5x
    
    return zoom


def create_stacked_frame(frame, people_data, output_width, output_height, aspect_ratio):
    """
    Create a frame with 2 people stacked based on aspect ratio.
    
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
            person_box = person['face_box'] or person['person_box']
            
            # Calculate section bounds
            y_start = i * section_height
            y_end = y_start + section_height
            if i == 1 and y_end < output_height:  # Last person gets any remaining pixels
                y_end = output_height
            actual_section_height = y_end - y_start
            actual_section_aspect = output_width / actual_section_height
            
            # Calculate optimal zoom for this person and section
            zoom = calculate_optimal_zoom(person_box, output_width, actual_section_height, frame_width, frame_height)
            
            # Crop person with dynamic zoom
            crop_box = calculate_crop_box(person_box, frame_width, frame_height, actual_section_aspect, zoom_factor=zoom)
            person_crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            # Resize using "cover" method (no stretching)
            person_resized = resize_cover(person_crop, output_width, actual_section_height)
            output_frame[y_start:y_end, :] = person_resized
    
    elif aspect_ratio > 1.2:  # Horizontal (16:9 = 1.778, landscape)
        # Stack horizontally (side by side)
        section_width = output_width // 2
        
        for i, person in enumerate(people_data):
            person_box = person['face_box'] or person['person_box']
            
            # Calculate section bounds
            x_start = i * section_width
            x_end = x_start + section_width
            if i == 1 and x_end < output_width:  # Last person gets any remaining pixels
                x_end = output_width
            actual_section_width = x_end - x_start
            
            # Each section maintains aspect ratio
            section_aspect = actual_section_width / output_height
            
            # Calculate optimal zoom for this person and section
            zoom = calculate_optimal_zoom(person_box, actual_section_width, output_height, frame_width, frame_height)
            
            # Crop person with dynamic zoom
            crop_box = calculate_crop_box(person_box, frame_width, frame_height, section_aspect, zoom_factor=zoom)
            person_crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            # Resize using "cover" method (no stretching)
            person_resized = resize_cover(person_crop, actual_section_width, output_height)
            output_frame[:, x_start:x_end] = person_resized
    
    else:  # Square-ish (1:1 = 1.0)
        # Side by side for 2 people
        section_width = output_width // 2
        section_aspect = section_width / output_height
        
        for i, person in enumerate(people_data):
            person_box = person['face_box'] or person['person_box']
            
            # Calculate optimal zoom for this person and section
            zoom = calculate_optimal_zoom(person_box, section_width, output_height, frame_width, frame_height)
            
            # Crop person with dynamic zoom
            crop_box = calculate_crop_box(person_box, frame_width, frame_height, section_aspect, zoom_factor=zoom)
            person_crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            
            # Resize using "cover" method (no stretching)
            person_resized = resize_cover(person_crop, section_width, output_height)
            
            x_start = i * section_width
            output_frame[:, x_start:x_start + section_width] = person_resized
    
    return output_frame


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


def process_video(input_video, final_output_video, model, face_cascade, aspect_ratio=9/16, analysis_scale=1.0, use_gpu=False, confidence_threshold=0.5):
    """
    Main video processing function that converts horizontal video to vertical format.
    
    Args:
        confidence_threshold: Minimum confidence (0-1) for person detection. Default: 0.5
    
    Args:
        input_video: Path to the input video file
        final_output_video: Path to the output video file
        model: YOLO model instance
        face_cascade: Haar Cascade face detector instance
        aspect_ratio: Target aspect ratio (width/height), default is 9/16 for vertical video
        analysis_scale: Scale factor for scene analysis (e.g., 0.5 = half resolution for faster processing)
        use_gpu: Whether GPU is available for hardware acceleration
    """
    script_start_time = time.time()
    
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

    print("\n🧠 Step 2: Analyzing scene content and determining strategy...")
    step_start_time = time.time()
    original_width, original_height = get_video_resolution(input_video)
    
    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * aspect_ratio)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    scenes_analysis = []
    for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing Scenes")):
        analysis = analyze_scene_content(input_video, start_time, end_time, model, face_cascade, analysis_scale, confidence_threshold=confidence_threshold)
        strategy, target_box = decide_cropping_strategy(analysis, original_height, original_width, aspect_ratio)
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box
        })
    step_end_time = time.time()
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

    print("\n📋 Step 3: Generated Processing Plan")
    for i, scene_data in enumerate(scenes_analysis):
        num_people = len(scene_data['analysis'])
        strategy = scene_data['strategy']
        start_time = scenes[i][0].get_timecode()
        end_time = scenes[i][1].get_timecode()
        
        # Calculate average confidence for this scene
        if num_people > 0:
            avg_confidence = sum(obj['confidence'] for obj in scene_data['analysis']) / num_people
            print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s) (avg conf: {avg_confidence:.2f}). Strategy: {strategy}")
        else:
            print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s). Strategy: {strategy}")

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
    
    # Pre-allocate letterbox frame to avoid repeated allocations
    letterbox_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    
    with tqdm(total=total_frames, desc="Applying Plan", smoothing=0.1) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_scene_index < len(scenes_analysis) - 1 and \
               frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                current_scene_index += 1

            scene_data = scenes_analysis[current_scene_index]
            strategy = scene_data['strategy']
            target_box = scene_data['target_box']

            if strategy == 'TRACK':
                crop_box = calculate_crop_box(target_box, original_width, original_height, aspect_ratio)
                processed_frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                output_frame = resize_cover(processed_frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            elif strategy == 'STACK':
                # Stack multiple people vertically
                people_data = target_box  # target_box contains the list of people for STACK strategy
                output_frame = create_stacked_frame(frame, people_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, aspect_ratio)
            else:  # LETTERBOX
                scale_factor = OUTPUT_WIDTH / original_width
                scaled_height = int(original_height * scale_factor)
                scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height), interpolation=cv2.INTER_LINEAR)
                
                # Reuse pre-allocated frame and clear it
                letterbox_frame.fill(0)
                y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
                letterbox_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
                output_frame = letterbox_frame
            
            # Write to FFmpeg pipe
            try:
                ffmpeg_process.stdin.write(output_frame.tobytes())
            except BrokenPipeError:
                print("\n⚠️  FFmpeg pipe broken, stopping...")
                break
                
            frame_number += 1
            pbar.update(1)
            
            # Explicit cleanup every 1000 frames to prevent memory buildup
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

