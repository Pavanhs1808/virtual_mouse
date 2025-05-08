# Eye Tracking Mouse Control

A Python-based eye tracking application that allows you to control your mouse cursor using eye movements and perform clicks using blinks.

## Features

- Control mouse cursor with eye movements
- Double-click using long blinks (hold for 1 second)
- Single-click using quick double blinks
- Real-time eye aspect ratio (EAR) visualization
- Smooth cursor movement with position stabilization
- Edge detection and screen boundary protection

## Requirements

- Python 3.10
- Webcam
- Windows OS

## Installation

1. Create a virtual environment:
```batch
python -m venv venv_py310
venv_py310\Scripts\activate
```

2. Install required packages:
```batch
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.7
pip install pyautogui==0.9.54
```

## Usage

1. Run the application:
```batch
python tracker.py
```

2. Position yourself in front of the webcam
3. Control the mouse cursor:
   - Look around to move the cursor
   - Long blink (1 second) for double-click
   - Quick double blink for single-click
4. Press 'q' to quit

## Controls

- **Cursor Movement**: Follow your eye gaze
- **Single Click**: Quick double blink
- **Double Click**: Long blink (hold for 1 second)
- **Exit**: Press 'q' key

## Configuration

Adjust these constants in `tracker.py` to customize behavior:

- `SMOOTHING_FACTOR`: Cursor movement smoothness (0.7 default)
- `SENSITIVITY`: Cursor movement speed (2.5 default)
- `BLINK_THRESHOLD`: Blink detection sensitivity (0.23 default)
- `LONG_BLINK_THRESHOLD`: Time for double-click (1.0 second default)
- `DOUBLE_BLINK_TIME`: Window for double blink detection (0.35 seconds default)

## Troubleshooting

- **Cursor too sensitive**: Decrease `SENSITIVITY` value
- **Clicks not registering**: Adjust `BLINK_THRESHOLD`
- **Movement too jerky**: Increase `SMOOTHING_FACTOR`
- **Accidental clicks**: Increase `BLINK_COOLDOWN`

## Dependencies

- OpenCV (cv2)
- MediaPipe
- PyAutoGUI
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
