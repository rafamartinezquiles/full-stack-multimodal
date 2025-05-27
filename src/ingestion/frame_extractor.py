# Import OpenCV for video processing
import cv2

# Import os for file and directory operations
import os

# Define a function to extract key frames from a video file
# Arguments:
# - video_path: path to the input video file
# - output_folder: where the extracted frames will be saved
# - every_n_seconds: interval (in seconds) at which to extract frames (default is every 5 seconds)
def extract_key_frames(video_path: str, output_folder: str, every_n_seconds: int = 5) -> list[str]:
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Load the video file using OpenCV's VideoCapture
    vidcap = cv2.VideoCapture(video_path)
    
    # Get the video's frame rate (frames per second)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on time (e.g., every 5 seconds)
    interval = int(fps * every_n_seconds)
    
    # List to store the paths of saved frame images
    frame_paths = []
    
    # Counters to track frame positions and saved images
    frame_idx = 0       # Index of the current frame being read
    saved_idx = 0       # Index used for naming saved frames
    
    # Loop through all frames in the video
    while True:
        success, frame = vidcap.read()  # Read the next frame
        if not success:
            break  # Stop if no more frames are available (end of video)

        # Save the frame if it's at the desired interval
        if frame_idx % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_idx}.jpg")
            cv2.imwrite(frame_filename, frame)  # Write the frame to an image file
            frame_paths.append(frame_filename)  # Track the saved frame path
            saved_idx += 1

        frame_idx += 1  # Move to the next frame

    # Release the video capture object to free resources
    vidcap.release()

    # Return the list of saved frame image paths
    return frame_paths
