import pyrealsense2 as rs
import os
import toml
import threading
import numpy as np
import cv2
from datetime import datetime
import shutil
import keyboard
import pandas as pd
import screeninfo
from Pose3D.calibration import calibrate_cams_all


# Get screen resolution automatically
screen = screeninfo.get_monitors()[0]
screen_width = screen.width
screen_height = screen.height

stop_recording = False  # Flag to stop threads
frames_dict = {}  # Store frames from each camera
lock = threading.Lock()  # Prevent race conditions


class CamThread(threading.Thread):
    def __init__(self, camera_serial):
        threading.Thread.__init__(self)
        self.camera_serial = camera_serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(camera_serial)
        self.pipeline.start(self.config)

    def run(self):
        global frames_dict
        print(f"Starting camera {self.camera_serial}")

        try:
            while not stop_recording:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                with lock:
                    frames_dict[self.camera_serial] = color_image

        finally:
            self.pipeline.stop()
            print(f"Stopped camera {self.camera_serial}")


def scale_images_to_fit(images, max_width, max_height):
    """Scales images to fit within the given width and height, preserving aspect ratio."""
    if not images:
        return None

    total_width = sum(img.shape[1] for img in images)
    max_img_height = max(img.shape[0] for img in images)

    scale_factor = min(max_width / total_width, max_height / max_img_height)

    resized_images = [
        cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
        for img in images
    ]

    return np.hstack(resized_images) if len(images) > 1 else resized_images[0]


def record_checkerboard():
    global stop_recording
    stop_recording = False

    ctx = rs.context()
    cameras_serial_numbers = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]

    if not cameras_serial_numbers:
        print("No Intel RealSense cameras detected.")
        return

    print(f"Detected cameras: {cameras_serial_numbers}")

    # Start camera threads
    threads = [CamThread(serial) for serial in cameras_serial_numbers]
    for thread in threads:
        thread.start()

    print("Press 's' to save an image pair and exit.")

    while True:
        with lock:
            images = [frames_dict[serial] for serial in cameras_serial_numbers if serial in frames_dict]

        if images:
            stacked_image = scale_images_to_fit(images, screen_width, screen_height)

            if stacked_image is not None:
                cv2.imshow("Camera View", stacked_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save images and exit
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_base_dir = os.path.join(os.getcwd(), "calibration", "extrinsics")

            with lock:
                for serial, img in frames_dict.items():
                    cam_dir = os.path.join(save_base_dir, serial)
                    os.makedirs(cam_dir, exist_ok=True)
                    save_path = os.path.join(cam_dir, f"cam_{serial}_{timestamp}.png")
                    cv2.imwrite(save_path, img)
                    print(f"Saved {save_path}")

            stop_recording = True
            break  # Exit loop after saving
        if key == ord('q'):
            stop_recording = True
            break

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()



def calculate_camera_matrix(intrinsics):
    # Calculate the camera matrix from intrinsics
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

def save_camera_parameters_to_toml(camera_params, file_path):
    with open(file_path, 'w') as f:
        toml.dump(camera_params, f)

def extract_intrinsics():
    ctx = rs.context()
    serial_numbers = []
    camera_params = {}  # Dictionary to hold all cameras' parameters

    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            serial_numbers.append(d.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
        return

    # Loop through all cameras
    for serial in serial_numbers:
        print(f"Extracting intrinsics for camera {serial}")

        # Start the pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)

        try:
            # Wait for a frame to get the intrinsics
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            # Calculate camera matrix and format parameters
            camera_matrix = calculate_camera_matrix(intrinsics)
            size = [float(val) for val in [intrinsics.width, intrinsics.height]]
            matrix = [[float(val) for val in row] for row in camera_matrix]

            # Prepare the camera's parameters dictionary
            camera_params[serial] = {
                "name": serial,
                "size": size,
                "matrix": matrix,
                "distortions": [0.0, 0.0, 0.0, 0.0],  # Assuming no distortion for simplicity
            }

        finally:
            # Stop the pipeline
            pipeline.stop()

    # Save all camera intrinsics to a single TOML file
    cwd = os.getcwd()
    calibration_dir = os.path.join(cwd, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    calib_file = os.path.join(calibration_dir, "Calib_board.toml")
    save_camera_parameters_to_toml(camera_params, calib_file)
    print(f"All camera intrinsics saved to {calib_file}")


def create_participant_data():
    # Define the headers for the Excel file
    headers = [
        "Participant ID", "Sex", "Age", "Weight", "Height", "Strong Leg"
    ]
    
    # Create a DataFrame with the specified headers
    df = pd.DataFrame(columns=headers)
    
    # Specify the Excel file name
    excel_file = "Participant_Data.xlsx"
    
    # Get the current working directory
    cwd = os.getcwd()
    
    # Save the DataFrame to an Excel file in the current working directory
    df.to_excel(os.path.join(cwd, excel_file), index=False)

# # Call the function to create the Excel file
# create_participant_data()

