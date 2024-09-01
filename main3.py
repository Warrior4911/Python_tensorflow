import cv2 as cv
import tensorflow as tf
import mediapipe as mp

# Ensure TensorFlow runs on the CPU.
tf.config.set_visible_devices([], 'GPU')

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open a video file or capture from a camera.
video_path = 'output_video_with_annotations.mp4'  # Replace with your video path
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Define the codec and create VideoWriter object.
fourcc = cv.VideoWriter.fourcc(*'XVID')
output_file = 'output_video.mp4'
out = cv.VideoWriter(output_file, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

if not out.isOpened():
    print(f"Error: Cannot open output file {output_file}")
    cap.release()
    exit()

print("Processing video... Press 'q' to quit the display window.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing the video.")
        break

    # Convert the frame to RGB.
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    results = hands.process(rgb_frame)

    # Blur the region containing the hand if detected.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box for the hand.
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            # Define the region of interest (ROI).
            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
            roi = frame[y_min:y_max, x_min:x_max]

            # Apply Gaussian Blur to the ROI.
            blurred_roi = cv.GaussianBlur(roi, (51, 51), 0)

            # Replace the ROI on the frame with the blurred version.
            frame[y_min:y_max, x_min:x_max] = blurred_roi

    # Write the frame with blurred hands to the output video file.
    out.write(frame)

    # Display the resulting frame.
    cv.imshow('Hand Blurring', frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release everything when done.
cap.release()
out.release()
cv.destroyAllWindows()

print(f"Processed video saved as {output_file}.")
