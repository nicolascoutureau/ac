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

def analyze_scene_content(video_path, scene_start_time, scene_end_time, model, face_cascade, analysis_scale=1.0, dnn_face_detector=None):
    """
    Analyzes the middle frame of a scene to detect people and faces.
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

    # Primary detection: YOLO for people
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:  # Person class
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

                detected_objects.append({'person_box': person_box, 'face_box': face_box})
    
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
            detected_objects.append({'person_box': person_box, 'face_box': face_box})
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


def decide_cropping_strategy(scene_analysis, frame_height, aspect_ratio):
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * aspect_ratio
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None


def calculate_crop_box(target_box, frame_width, frame_height, aspect_ratio):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * aspect_ratio)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def process_video(input_video, final_output_video, model, face_cascade, aspect_ratio=9/16, analysis_scale=1.0):
    """
    Main video processing function that converts horizontal video to vertical format.
    
    Args:
        input_video: Path to the input video file
        final_output_video: Path to the output video file
        model: YOLO model instance
        face_cascade: Haar Cascade face detector instance
        aspect_ratio: Target aspect ratio (width/height), default is 9/16 for vertical video
        analysis_scale: Scale factor for scene analysis (e.g., 0.5 = half resolution for faster processing)
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
        analysis = analyze_scene_content(input_video, start_time, end_time, model, face_cascade, analysis_scale)
        strategy, target_box = decide_cropping_strategy(analysis, original_height, aspect_ratio)
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
        print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s). Strategy: {strategy}")

    print("\n✂️ Step 4: Processing video frames...")
    step_start_time = time.time()
    
    # Use ultrafast preset to prevent FFmpeg from becoming the bottleneck
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-crf', '23', '-an', temp_video_output
    ]

    # Large buffer to prevent pipe deadlocks
    ffmpeg_process = subprocess.Popen(
        command, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8  # 100MB buffer
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
                output_frame = cv2.resize(processed_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
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

