import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Initialize Mediapipe drawing and pose instances
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points (for pose estimation)
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Last point

    # Calculate the angle in radians and convert to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is within 0-180 degrees
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to start pose estimation
def start_estimation(event):
    global estimation_running
    estimation_running = True
    run_pose_estimation()

# Function to exit the application
def exit_app(event):
    plt.close()

# Function to run pose estimation
def run_pose_estimation():
    global estimation_running, counter, stage
    cap = cv2.VideoCapture(0)

    while estimation_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture video.")
            break

        # Recolor the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame for pose estimation
        results = pose.process(image)

        # Recolor back to BGR for displaying
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract pose landmarks if available
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for shoulder, elbow, and wrist for angle calculation
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle at the elbow
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize the angle on the image
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(f"Curl Count: {counter}")
        else:
            print("No pose landmarks detected.")

        # Render the curl counter on the screen
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Display the number of repetitions
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the stage (up or down)
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage if stage else "N/A",
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display using Matplotlib
        ax.clear()  # Clear previous image
        ax.imshow(image)  # Show the current frame
        ax.axis('off')  # Hide axes
        plt.pause(0.001)  # Pause to allow the image to be displayed

    cap.release()

# Initialize variables
estimation_running = False
counter = 0
stage = None

# Setup Mediapipe Pose instance
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup Matplotlib figure and buttons
plt.ion()  # Enable interactive mode for Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))  # Create a Matplotlib figure and axis

# Create buttons for starting and exiting the application
button_ax_start = plt.axes([0.3, 0.07, 0.4, 0.05])  # Position for the start button above the exit button
start_button = Button(button_ax_start, 'Start Pose Estimation')
start_button.on_clicked(start_estimation)

button_ax_exit = plt.axes([0.3, 0.01, 0.4, 0.05])  # Position for the exit button below the start button
exit_button = Button(button_ax_exit, 'Exit')
exit_button.on_clicked(exit_app)

# Create a title above the buttons using text
plt.text(0.5, 0.95, "Pose Recognition", fontsize=16, ha='center', va='center', transform=fig.transFigure)

plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust the top and bottom margins of the plot
plt.show()  # Show the plot with buttons
plt.axis('off')  # Hide the axes
plt.show(block=True)  # Keep the plot open
