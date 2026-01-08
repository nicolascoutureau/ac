# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import cv2
from ultralytics import YOLO
from autocrop import process_video, AUTOCROP_VERSION
import tempfile
import os
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print(f"🚀 Autocrop Predictor v{AUTOCROP_VERSION}")
        print(f"=" * 40)
        
        try:
            # Detect hardware acceleration availability
            self.has_gpu = torch.cuda.is_available()
            self.device = 'cuda:0' if self.has_gpu else 'cpu'
            
            if self.has_gpu:
                print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("💻 No GPU detected, using CPU")
                # Enable CPU optimizations
                num_threads = os.cpu_count() or 4
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(num_threads)
                print(f"  Using {num_threads} CPU threads")
            
            # Pre-load different model sizes for different speed presets
            print(f"Loading YOLO models on {self.device}...")
            self.models = {
                'quality': YOLO('yolov8x.pt'),  # 131 MB, most accurate
                'balanced': YOLO('yolov8m.pt'),  # 49 MB, good balance
                'fast': YOLO('yolov8s.pt')       # 21 MB, fast
            }
            
            # Move models to GPU if available
            for name, model in self.models.items():
                model.to(self.device)
            
            print(f"  ✅ YOLOv8x (quality), YOLOv8m (balanced), YOLOv8s (fast) loaded on {self.device}")
            
            # Load YOLO face model for accurate face detection
            self.face_model = None
            script_dir = os.path.dirname(os.path.abspath(__file__))
            face_model_paths = [
                os.path.join(script_dir, "yolov8n-face.pt"),  # Same directory as predict.py (/src/ on Replicate)
                "/src/yolov8n-face.pt",  # Replicate container /src/
                "/root/yolov8n-face.pt",  # Cog build default directory
                "/yolov8n-face.pt",  # Root directory (Cog may download here)
                "yolov8n-face.pt",  # Current working directory
            ]
            
            print(f"Looking for YOLO face model (script_dir: {script_dir})...")
            for face_model_path in face_model_paths:
                print(f"  Checking: {face_model_path} ... ", end="")
                if os.path.exists(face_model_path):
                    # Check file size - yolov8n-face.pt should be ~6MB
                    file_size = os.path.getsize(face_model_path)
                    print(f"FOUND! ({file_size / 1024 / 1024:.2f} MB)")
                    if file_size < 1_000_000:  # Less than 1MB is likely corrupted
                        print(f"  ⚠️  File too small, likely corrupted (expected ~6MB). Skipping.")
                        continue
                    try:
                        print(f"Loading YOLO face model from {face_model_path}...")
                        self.face_model = YOLO(face_model_path)
                        self.face_model.to(self.device)
                        print(f"  ✅ YOLOv8n-face loaded on {self.device}")
                        break
                    except Exception as e:
                        print(f"  ⚠️  Failed to load face model: {e}. Trying next path...")
                        self.face_model = None
                        continue
                else:
                    print("not found")
            
            if self.face_model is None:
                print(f"  ⚠️  YOLO face model not available, will use face estimation from person boxes")
            
            print("Loading Haar Cascade face detector...")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar Cascade classifier")
            
            print("  Haar Cascade loaded successfully")
            print("✅ Setup complete!")
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            raise

    def predict(
        self,
        video: Path = Input(description="Input horizontal video to convert to vertical format"),
        aspect_ratio: str = Input(
            description="Output aspect ratio",
            choices=["9:16", "1:1", "16:9"],
            default="9:16"
        ),
        speed_preset: str = Input(
            description="Processing speed preset. Fast uses smaller YOLO model and lower resolution analysis.",
            choices=["quality", "balanced", "fast"],
            default="balanced"
        ),
        detect_speaker: bool = Input(
            description="Detect and focus on the active speaker using fast lip tracking. When multiple people are detected, focuses on who is talking.",
            default=False
        ),
        tracking_mode: str = Input(
            description="Camera tracking mode. Smooth = cinematic OpusClip-like movement. Static = fixed per-scene.",
            choices=["smooth", "static", "fast"],
            default="smooth"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        output_path = None
        
        try:
            print(f"\n🎥 Starting video processing...")
            print(f"  Input: {video}")
            print(f"  Aspect ratio: {aspect_ratio}")
            print(f"  Speed preset: {speed_preset}")
            print(f"  Speaker detection: {'MediaPipe (fast)' if detect_speaker else 'Disabled'}")
            print(f"  Tracking mode: {tracking_mode}")
            
            # Select model based on speed preset
            model = self.models[speed_preset]
            model_names = {
                'quality': 'YOLOv8x (slowest, most accurate)',
                'balanced': 'YOLOv8m (3x faster, still accurate)',
                'fast': 'YOLOv8s (5x faster, good accuracy)'
            }
            print(f"  Using: {model_names[speed_preset]}")
            
            # Set analysis resolution based on preset
            # With L40S GPUs, we can afford higher resolution analysis
            analysis_scale = {
                'quality': 1.0,    # Full resolution
                'balanced': 1.0,   # Full resolution (L40S can handle it)
                'fast': 0.75       # 75% resolution for analysis
            }
            
            # Create a temporary output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
                output_path = temp_output.name
            
            # Parse aspect ratio string to numeric value
            ratio_parts = aspect_ratio.split(':')
            aspect_ratio_numeric = int(ratio_parts[0]) / int(ratio_parts[1])
            print(f"  Numeric aspect ratio: {aspect_ratio_numeric:.4f}")
            
            # Verify video file exists and is readable
            if not os.path.exists(str(video)):
                raise FileNotFoundError(f"Input video not found: {video}")
            
            # Process the video
            process_video(
                input_video=str(video),
                final_output_video=output_path,
                model=model,
                face_cascade=self.face_cascade,
                aspect_ratio=aspect_ratio_numeric,
                analysis_scale=analysis_scale[speed_preset],
                use_gpu=self.has_gpu,
                detect_speaker=detect_speaker,
                tracking_mode=tracking_mode,
                face_model=self.face_model  # Pass pre-loaded face model
            )
            
            # Verify output was created
            if not os.path.exists(output_path):
                raise RuntimeError("Output video file was not created")
            
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n✅ Processing complete! Output size: {output_size:.2f} MB")
            
            return Path(output_path)
            
        except Exception as e:
            print(f"\n❌ Error during prediction: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            # Clean up on error
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Video processing failed: {str(e)}") from e
