
# Pose Recognition App
This Python application leverages OpenCV and MediaPipe to perform real-time pose estimation using a webcam. It detects specific body angles and counts movements, making it suitable for applications like exercise tracking. The app includes an interactive graphical interface built with Matplotlib, allowing users to start and stop pose recognition sessions.

Features: 
* Real-time pose estimation and tracking with MediaPipe
* Angle calculation between joints for accurate movement detection
* Curl counter and movement stage tracking (e.g., "up" and "down" positions)
* User interface with start and exit buttons for easy control

# Installation
To run this app, you'll need to install the following Python libraries. Open a command prompt and enter each of these commands:

pip install opencv-python  
pip install mediapipe  
pip install numpy  
pip install matplotlib  

#Requirements
Python 3.7+
A webcam for real-time pose estimation

#Notes
This project demonstrates the basics of pose estimation using machine learning models for joint detection. It can be extended with more complex movement analysis or adapted for other applications involving human motion tracking.
