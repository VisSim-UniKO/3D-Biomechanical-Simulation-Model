import os
import cv2
from natsort import natsorted

# enumerate frames in a directory (optionally from start_frame to end_frame)
def frame_enumerator(dir_path, start_frame=None, end_frame=None):
    frame_files = natsorted(
        [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )
    for i, frame_file in enumerate(frame_files):
        if start_frame is not None and i < start_frame:
            continue
        if end_frame is not None and i > end_frame:
            break
        frame_path = os.path.join(dir_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            yield i, frame, frame_file


def save_cropped_frames(input_source, output_dir, start_frame=None, end_frame=None):
    """
    Loops through frames from input_source, crops each frame, and saves them to the output directory.
    Args:
        input_source (str): Path to the directory containing the input frames.
        output_dir (str): Path to the directory where cropped frames will be saved.
        start_frame (int, optional): The frame index to start processing (inclusive).
        end_frame (int, optional): The frame index to stop processing (inclusive).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over frames from the input source using the frame_enumerator
    for i, frame, frame_file in frame_enumerator(input_source, start_frame=start_frame, end_frame=end_frame):
        # Crop the frame (from column 300 to 700)
        cropped_frame = frame[:, 300:700]

        # Construct the full output path
        output_path = os.path.join(output_dir, frame_file)

        # Save the cropped frame
        cv2.imwrite(output_path, cropped_frame)
        print(f"Saved cropped frame {frame_file} to {output_path}")




# cwd = os.getcwd()
# data_dir = os.path.join(cwd, "src/data")
# participants = ["A", "B", "C", "D", "E", "I", "J", "K", "L", "M", "N", "O"]

# for participant in participants:
#     participant_dir = participant + "-squat"
#     input_path = os.path.join(data_dir, "videos", participant_dir, "videos/244622073485_right")
#     output_path = os.path.join(data_dir, "videos_cropped", participant_dir)
#     os.makedirs(output_path, exist_ok=True)
#     save_cropped_frames(input_path, output_path)
