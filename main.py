import cv2
import numpy as np

# Define named constants for magic numbers
FRAME_DIFFERENCE = 15000
BINARY_THRESHOLD = 25

def is_frame_stable(frame1: np.ndarray, frame2: np.ndarray, threshold: int = FRAME_DIFFERENCE):

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between frames
    diff = cv2.absdiff(gray1, gray2)

    # Apply a binary threshold to get binary image
    _, diff = cv2.threshold(diff, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Calculate the number of non-zero (white) pixels in the difference image
    count_non_zero = np.count_nonzero(diff)

    # Check if the difference is less than the threshold
    return count_non_zero < threshold

def process_video(input_path: str, output_path: str, frame_difference_threshold: int = FRAME_DIFFERENCE) :

# Processes the video to remove frames where pages are being turned.

    try:
        # Open the video file
        cap = cv2.VideoCapture("Source_Video/raw_input_video.mp4")
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return
        
        # Get the width, height, and FPS of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define the codec and create a VideoWriter object to write the output video
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Read the first frame
        success, prev_frame = cap.read()
        
        if not success:
            print("Error reading the first frame")
            return
        
        # Write the first frame assuming it's stable (this can be adjusted based on context)
        out.write(prev_frame)

        while True:
            # Read the next frame
            success, curr_frame = cap.read()
            
            if not success:
                break

            # Check if the current frame is stable
            if is_frame_stable(prev_frame, curr_frame, frame_difference_threshold):
                # If stable, write the frame to the output video
                out.write(curr_frame)
            
            # Update the previous frame
            prev_frame = curr_frame

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release video objects
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        print("Processing completed. The output video has been saved.")

# Example usage
input_videopath = 'Source_Video/raw_input_video.mp4'  # Path to the input video file
output_video = 'input_video.mp4'  # Path to save the output video file
process_video(input_videopath, output_video)

