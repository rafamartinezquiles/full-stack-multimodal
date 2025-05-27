# Import OpenCV for video processing
import cv2

# Import os for file and directory operations
import os

# Define a function to extract key frames from a video file
# Arguments:
# - video_path: path to the input video file
# - output_folder: where the extracted frames will be saved
# - every_n_seconds: interval (in seconds) at which to extract frames 
def extract_key_frames(video_path: str, output_folder: str, every_n_seconds: int = 5) -> list[str]:
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Load the video file using OpenCV's VideoCapture
    vidcap = cv2.VideoCapture(video_path)
    
    # Get the video's frame rate 
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on time 
    interval = int(fps * every_n_seconds)
    
    # List to store the paths of saved frame images
    frame_paths = []
    
    # Counters to track frame positions and saved images
    frame_idx = 0 
    saved_idx = 0
    
    # Loop through all frames in the video
    while True:
        success, frame = vidcap.read() 
        if not success:
            break  

        # Save the frame if it's at the desired interval
        if frame_idx % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_idx}.jpg")
            cv2.imwrite(frame_filename, frame)  
            frame_paths.append(frame_filename)  
            saved_idx += 1

        frame_idx += 1  

    # Release the video capture object to free resources
    vidcap.release()

    # Return the list of saved frame image paths
    return frame_paths
