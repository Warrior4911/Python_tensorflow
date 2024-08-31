import cv2
import numpy as np
import tensorflow as tf

# Parameters for optical flow
MOTION_THRESHOLD = 0.5  # Adjust this based on your use case
TEXT_POSITION = (10, 30)  # Position for the text on the frame

# Load TensorFlow model (optional, for additional analysis)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    batch_frame = np.expand_dims(normalized_frame, axis=0)
    return batch_frame

def detect_paper_turn(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    # Example: Use predictions to adjust the logic if needed
    return predictions

# Load video file
cap = cv2.VideoCapture('input_video.mp4')

# Check if video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
output_file = 'output_video_with_annotations.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi files
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Parameters for optical flow
previous_magnitude_sum = 0
direction = "Stable"

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Calculate average magnitude of optical flow
    avg_magnitude = np.mean(magnitude)
    
    # Determine motion direction based on magnitude change
    if avg_magnitude > previous_magnitude_sum + MOTION_THRESHOLD:
        direction = "Paper Turning: Right to Left"
    elif avg_magnitude < previous_magnitude_sum - MOTION_THRESHOLD:
        direction = "Paper Turning: Left to Right"
    else:
        direction = "Stable"
    
    previous_magnitude_sum = avg_magnitude

    # Display the frame difference and optical flow magnitude
    cv2.imshow('Frame Difference', thresh)
    
    # Normalize the optical flow magnitude for display
    flow_display = np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow('Optical Flow', flow_display)


    # Display text indicating the direction of paper turning
    frame_with_text = frame2.copy()
    cv2.putText(frame_with_text, direction, TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video with Direction', frame_with_text)
    
    # Save the annotated frame to the output video
    out.write(frame_with_text)

    # Optionally, use model detection on the current frame
    predictions = detect_paper_turn(frame2)
    print(predictions)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update previous frame
    prev_gray = gray

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
