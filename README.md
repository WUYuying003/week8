# 😊 笑容收集之旅 (Smile Collection Journey)# MediaPipe Examples - Week 08



An interactive hand gesture game inspired by NEX HomeCourt fitness training games.This folder contains comprehensive examples of using Google's MediaPipe framework for computer vision and machine learning tasks.



![Game Preview](https://img.shields.io/badge/Python-3.11-blue)## Overview

![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-green)

![OpenCV](https://img.shields.io/badge/OpenCV-Latest-orange)MediaPipe is a powerful framework for building multimodal applied ML pipelines. These examples demonstrate various computer vision capabilities including face detection, hand tracking, pose estimation, and more.



## 🎮 Game Overview## Requirements



**笑容收集之旅** is a motion-based fitness game where players use hand gestures to interact with on-screen targets. Touch numbered circles in sequential order to collect smiling faces and advance through increasingly challenging levels!Install the required packages:



### Key Features```bash

pip install -r requirements.txt

- 🖐️ **Dual Hand Tracking**: Supports both left and right hand interactions using MediaPipe```

- 🎯 **Sequential Gameplay**: Touch numbered targets in order (1→2→3→...)

- ⏱️ **Countdown Progress Bars**: Each target has a 10-second countdown timer displayed as a circular progress bar## Examples

- 😊 **Flying Smiley Animations**: Collected smiles fly to the score area with smooth animations

- 🎚️ **Progressive Difficulty**: Start with 2 targets, gain 1 more each level (max 8)### 1. Face Detection (`1_face_detection.py`)

- 👆 **Touch-based UI**: "NEXT" button can be touched with hand gestures to advance levels- Basic face detection using MediaPipe

- ❌ **Error Detection**: Wrong touch triggers visual feedback and level restart- Draws bounding boxes around detected faces

- 🎨 **Visual Feedback**: - Shows confidence scores

  - Current target: White with thick border- Good starting point for understanding MediaPipe basics

  - Waiting targets: Blue hollow circles

  - Completed targets: Blue filled circles**Key Features:**

  - Wrong targets: Red with X mark- Real-time face detection

- Confidence scoring

## 🚀 Quick Start- Bounding box visualization



### Prerequisites### 2. Hand Tracking (`2_hand_tracking.py`)

- Detects and tracks hand landmarks

- Python 3.11+- Identifies fingertips with blue dots

- Webcam- Works with both hands simultaneously

- macOS/Windows/Linux- 21 landmarks per hand



### Installation**Key Features:**

- 21-point hand landmark detection

1. Clone the repository:- Fingertip identification

```bash- Multi-hand support

git clone https://github.com/WUYuying003/week8.git- Real-time tracking

cd week8

```### 3. Pose Estimation (`3_pose_estimation.py`)

- Full body pose detection with 33 landmarks

2. Install required packages:- Calculates joint angles (demonstrated with left arm)

```bash- Detects specific poses (hands up detection)

pip install -r requirements.txt- Body keypoint connections

```

**Key Features:**

3. Configure your camera:- 33-point pose landmarks

```bash- Joint angle calculation

python setup_camera.py- Pose recognition

```- Real-time body tracking



4. Run the game:### 4. Face Mesh (`4_face_mesh.py`)

```bash- Detailed face landmark detection with 468 points

python fitness_touch_game.py- Multiple visualization modes: contours, full mesh, irises

```- High-precision facial feature mapping

- Interactive mode switching

## 📖 How to Play

**Key Features:**

1. **Start the game**: Run `fitness_touch_game.py`- 468 facial landmarks

2. **Position yourself**: Stand in front of your webcam with good lighting- Face contours visualization

3. **Raise your hand**: Use your index finger to interact with targets- Full face mesh tessellation

4. **Touch in order**: Touch the numbered circles sequentially (1→2→3→...)- Iris tracking (when enabled)

5. **Collect smiles**: Each correct touch earns you a smiley face 😊- Interactive controls to switch between modes

6. **Advance levels**: After completing all targets, touch the "NEXT" button

7. **Challenge yourself**: Each level adds one more target to touch!**Controls:**

- `c` - Contours only mode

### Controls- `f` - Full mesh mode  

- `i` - Irises + contours mode

- **ESC**: Quit game

- **R**: Restart game### 5. Gesture Recognition (`5_gesture_recognition.py`)

- **SPACE**: Alternative way to advance to next level- Recognizes common hand gestures

- **Hand Gestures**: Touch targets and buttons with your index finger- Supports thumbs up/down, peace sign, rock on, numbers

- Real-time gesture classification

## 🎯 Game Rules- Works with both hands



- You must touch targets in the correct numerical order (1→2→3→...)**Supported Gestures:**

- Each target has a 10-second countdown timer- 👍 Thumbs Up

- Touching the wrong target results in level restart- 👎 Thumbs Down

- Complete all targets before time runs out- ✌️ Peace Sign

- Smiley animations must finish before advancing to next level- 🤟 Rock On

- ✊ Fist

## 📁 Project Structure- 🖐️ Open Hand

- 👉 Pointing

```- 👌 OK Sign

week8/- Numbers 1-5

├── fitness_touch_game.py      # Main game file (笑容收集之旅)

├── camera_utils.py            # Camera setup utilities### 6. Holistic Detection (`6_holistic_detection.py`)

├── setup_camera.py            # Camera configuration tool- Combines face mesh, pose, and hand detection

├── requirements.txt           # Python dependencies- Unified model for comprehensive body analysis

├── README.md                  # This file- Shows detection status for each component

├── 1_face_detection.py        # MediaPipe face detection example- Counts total landmarks detected

├── 2_hand_tracking.py         # MediaPipe hand tracking example

├── 3_pose_estimation.py       # MediaPipe pose estimation example**Key Features:**

├── 4_face_mesh.py            # MediaPipe face mesh example- Face mesh (468 landmarks)

├── 5_gesture_recognition.py   # Gesture recognition example- Pose detection (33 landmarks)

├── 6_holistic_detection.py    # Holistic detection example- Hand detection (21 landmarks each)

├── 7_selfie_segmentation.py   # Selfie segmentation example- Integrated processing

└── 8_multi_detection.py       # Multi-detection example

```### 7. Selfie Segmentation (`7_selfie_segmentation.py`)

- Person segmentation for virtual backgrounds

## 🛠️ Technical Details- Multiple background effects (solid colors, gradient, patterns)

- Real-time background replacement

### Technologies Used- Adjustable segmentation threshold



- **Python 3.11**: Main programming language**Background Effects:**

- **MediaPipe 0.10.21**: Hand tracking and pose detection- Solid colors (blue, green, red)

- **OpenCV**: Computer vision and image processing- Gradient backgrounds

- **NumPy**: Numerical computations- Checkerboard pattern

- Original view toggle

### Game Architecture

### 8. Multi-Detection System (`8_multi_detection.py`)

The game is built with a class-based architecture (`FitnessTouchGame`) that handles:- Combines multiple MediaPipe models

- Hand tracking via MediaPipe- Toggle individual detection modules

- Game state management (playing, level complete, wrong touch, time up)- Performance monitoring with FPS counter

- Target generation and positioning- Comprehensive body analysis system

- Collision detection for finger-target interactions

- Animation systems for flying smileys**Features:**

- UI rendering and visual feedback- Face mesh detection

- Hand tracking

### Key Components- Pose estimation

- Background segmentation

1. **Hand Tracking**: Uses MediaPipe Hands solution to detect index finger tip position- Real-time performance metrics

2. **Target System**: Randomly generates non-overlapping targets with wider screen distribution- Modular activation/deactivation

3. **Progress Bars**: Circular countdown timers around each active target

4. **Animation Engine**: Smooth interpolation for flying smiley faces## 📷 Camera Setup

5. **State Machine**: Manages game states and transitions

### First Time Setup

## 🎨 CustomizationBefore running any MediaPipe examples, configure your camera:



You can modify game parameters in `fitness_touch_game.py`:```bash

python setup_camera.py

```python```

self.target_radius = 50              # Size of target circles

self.touch_threshold = 60            # Distance for successful touchThis will:

self.target_countdown_duration = 10.0  # Seconds per target1. 🔍 Detect all available cameras (tests devices 0-10)

self.targets_per_level = ...         # Number of targets formula2. 📋 Show you working cameras with their resolutions and FPS

```3. 🎥 Let you test the selected camera with live preview

4. 💾 Save the working camera ID to `.env` file

## 🐛 Troubleshooting5. ✅ All other scripts will automatically use the configured camera



### Camera Issues### Camera Configuration Details

- Run `python setup_camera.py` to configure your camera

- Grant camera permissions in System Preferences (macOS)The setup script creates a `.env` file with your camera device ID:

- Ensure no other apps are using the camera```

CAMERA_DEVICE=4

### Hand Detection Issues```

- Ensure good lighting conditions

- Position yourself 1-2 meters from the cameraAll MediaPipe scripts automatically load this configuration using `camera_utils.py`.

- Keep your hand clearly visible and unobstructed

### Manual Camera Override

### Performance Issues

- Close other camera-using applicationsIf you need to use a different camera temporarily, you can:

- Reduce video resolution if needed1. Edit the `.env` file directly

- Ensure GPU drivers are up to date2. Run `setup_camera.py` again to reconfigure

3. Delete `.env` file to force auto-detection

## 📝 License

## Usage Tips

This project is created for educational purposes as part of SD5913 - Programming for Art and Design.

### Camera Best Practices

## 🙏 Acknowledgments- 🚫 Close other camera applications (Zoom, Skype, etc.)

- 💡 Ensure good lighting for better detection accuracy

- Inspired by [NEX HomeCourt](https://www.homecourt.ai/) fitness training games- 📏 Position yourself 1-2 meters from the camera for optimal results

- Built with [Google MediaPipe](https://mediapipe.dev/)- 🔄 If camera stops working, run `setup_camera.py` to reconfigure

- Course: SD5913 - Programming for Art and Design

### Performance Optimization

## 👤 Author- Close other camera applications

- Reduce video resolution if experiencing lag

Created by WUYuying003- Toggle off unused detection modules in multi-detection example



---

### Camera Issues

**Enjoy collecting smiles!** 😊🎮✨```python

# Check available cameras
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
```

## Controls

### Common Controls (Most Examples)
- `q` - Quit application
- `ESC` - Alternative quit key

### Specific Controls
- **Face Mesh**: `c` - Contours, `f` - Full mesh, `i` - Irises
- **Selfie Segmentation**: `b` - Change background, `o` - Original view
- **Multi-Detection**: `f` - Face, `h` - Hands, `p` - Pose, `s` - Segmentation

## Applications

These examples can be used as building blocks for:

- **Fitness Applications**: Pose estimation for exercise tracking
- **Sign Language Recognition**: Hand gesture classification
- **Video Conferencing**: Background replacement and effects
- **Augmented Reality**: Face mesh and hand tracking for AR filters
- **Security Systems**: Face detection and recognition
- **Accessibility Tools**: Gesture-based control interfaces
- **Gaming**: Motion-based game controls
- **Medical Applications**: Posture analysis and rehabilitation
- **Beauty Applications**: Face mesh for makeup filters and facial analysis

## Performance Notes

- **Face Detection**: ~30-60 FPS on most modern computers
- **Hand Tracking**: ~20-40 FPS depending on number of hands
- **Pose Estimation**: ~15-30 FPS for full body tracking
- **Face Mesh**: ~20-40 FPS depending on visualization mode
- **Gesture Recognition**: ~25-40 FPS with hand tracking overhead
- **Holistic Model**: ~10-25 FPS (combines all features)
- **Selfie Segmentation**: ~20-35 FPS depending on background complexity
- **Multi-Detection**: Varies based on active modules

## Further Reading

- [MediaPipe Documentation](https://mediapipe.dev/)

## Advanced Usage

For production applications, consider:
- Model optimization for specific use cases
- Custom training for specialized gestures
- Integration with other ML frameworks
- Mobile deployment using MediaPipe mobile solutions
