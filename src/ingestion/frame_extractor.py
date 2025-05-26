import cv2
import os

def extract_key_frames(video_path: str, output_folder: str, every_n_seconds: int = 5) -> list[str]:
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * every_n_seconds)
    
    frame_paths = []
    frame_idx = 0
    saved_idx = 0
    
    while True:
        success, frame = vidcap.read()
        if not success:
            break

        if frame_idx % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_idx}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
            saved_idx += 1

        frame_idx += 1

    vidcap.release()
    return frame_paths
