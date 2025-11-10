# Autocrop-Vertical Replicate Deployment Guide

## Overview

This project converts horizontal videos to vertical format (9:16, 1:1, or 16:9) using intelligent content-aware cropping with YOLOv8x object detection.

## Features

✅ **Aspect Ratio Options**: 9:16 (vertical), 1:1 (square), 16:9 (horizontal)
✅ **Speed Presets**: Quality, Balanced (3x faster), Fast (5x faster)
✅ **Smart Detection**: YOLOv8 models (x/m/s) for person detection + Haar Cascade for faces
✅ **Fallback Detection**: Haar Cascade fallback when YOLO doesn't detect people
✅ **CPU Optimized**: Multi-threaded inference with model fusion
✅ **Pre-downloaded Models**: All YOLO models downloaded during Docker build

## Local Testing

### 1. Build the container
```bash
cog build
```

This will:
- Install all dependencies
- Download YOLOv8x model (131 MB) - happens once during build
- Set up FFmpeg

### 2. Test a prediction
```bash
cog predict -i video=@your_video.mp4 -i aspect_ratio="9:16"
```

### 3. Test different aspect ratios and speed presets
```bash
# Vertical (9:16) - default for TikTok, Reels, Shorts
cog predict -i video=@input.mp4 -i aspect_ratio="9:16"

# Square (1:1) - Instagram posts
cog predict -i video=@input.mp4 -i aspect_ratio="1:1"

# Horizontal (16:9) - Standard widescreen
cog predict -i video=@input.mp4 -i aspect_ratio="16:9"

# Speed presets (balanced is default)
cog predict -i video=@input.mp4 -i speed_preset="quality"   # Slowest, most accurate
cog predict -i video=@input.mp4 -i speed_preset="balanced"  # 3x faster, good accuracy
cog predict -i video=@input.mp4 -i speed_preset="fast"      # 5x faster, decent accuracy
```

## Deployment to Replicate

### 1. Push to Replicate
```bash
cog push r8.im/your-username/autocrop-vertical
```

### 2. Set Model Settings
On Replicate dashboard:
- **Hardware**: CPU (default)
- **Visibility**: Public or Private
- **Timeout**: Default should work now (model is pre-downloaded)

### 3. Use via API
```python
import replicate

output = replicate.run(
    "your-username/autocrop-vertical",
    input={
        "video": open("input.mp4", "rb"),
        "aspect_ratio": "9:16"
    }
)
print(output)
```

## How It Works

1. **Scene Detection**: PySceneDetect splits video into scenes
2. **Content Analysis**: YOLOv8x detects people, Haar Cascade detects faces
3. **Strategy Decision**: 
   - **TRACK**: Crops to follow detected people/faces
   - **LETTERBOX**: Adds black bars when people are too spread out
4. **Processing**: FFmpeg efficiently encodes the final video
5. **Audio**: Original audio is preserved

## Troubleshooting

### Build takes a long time
- Normal! YOLOv8x is 131 MB and needs to download
- Only happens once during build, not at runtime

### Out of Memory
- Try shorter videos first (< 1 minute)
- Consider using a smaller model (edit `cog.yaml` run command to use `yolov8m.pt`)

### Processing is slow
- CPU-only inference is slow by design
- YOLOv8x on CPU can take 2-3x video duration to process
- For faster results, switch to `yolov8m.pt` or `yolov8s.pt`

### Prediction fails with "PA" error
- Should be fixed with pre-download
- If persists, check Replicate logs for specific error

## Model Comparison

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| yolov8n | 6.2 MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Quick tests |
| yolov8s | 21.5 MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Fast production |
| yolov8m | 49.7 MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Balanced |
| yolov8x | 131 MB | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | Best quality |

To change model, edit line 21 in `cog.yaml` and line 25 in `predict.py`.

## Performance Expectations

### Processing Speed by Preset (CPU)

| Preset | Model | Resolution | 51min Video | 5min Video | Best For |
|--------|-------|------------|-------------|------------|----------|
| **Quality** | YOLOv8x | 100% | ~15 min | ~1.5 min | Maximum accuracy |
| **Balanced** | YOLOv8m | 75% | **~5 min** | **~30 sec** | Best balance ⭐ |
| **Fast** | YOLOv8s | 50% | **~3 min** | **~18 sec** | Quick results |

- **Setup**: < 10 seconds (all models pre-downloaded)
- **Default**: Balanced preset (3x faster than quality)

## Files Overview

- `predict.py` - Replicate prediction interface
- `autocrop.py` - Core video processing logic
- `cog.yaml` - Container configuration
- `requirements.txt` - Python dependencies
- `.dockerignore` - Files to exclude from container

## Credits

Based on [Autocrop-vertical](https://github.com/kamilstanuch/Autocrop-vertical) by Kamil Stanuch

