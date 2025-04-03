import threading
import os
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
from glob import glob
import keyboard

def get_run_number(id, posture):
    recording_dir = f"./{id}-{posture}"  
    folders = glob(f"{recording_dir}-*/", recursive=False)
    print(f'Folders found for {id}-{posture}:', folders)
    return len(folders) + 1

class camThread(threading.Thread):
    def __init__(self, cameraSerialNum, tp, pos, recording_dir):
        threading.Thread.__init__(self)
        self.cameraSerialNumber = cameraSerialNum
        self.tp = tp
        self.pos = pos
        self.dir = recording_dir

    def run(self):
        print("Starting camera=" + str(self.cameraSerialNumber))
        cameraRun(self.cameraSerialNumber, self.tp, self.pos, self.dir)

def cameraRun(cameraId, tp, posture, recording_dir):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(cameraId)

    pipeline_profile = config.resolve(pipeline)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(f'device_product_line: {device_product_line}')

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    depth_dir = os.path.join(recording_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_sensor.set_option(rs.option.visual_preset, 3)

    # Write depth scale
    depth_scale_file = os.path.join(depth_dir, f"camId_{cameraId}_depthScale.txt")
    with open(depth_scale_file, "w") as f:
        f.write(f'depth_scale: {depth_scale}')

    i = 0
    try:
        while not stop_recording:
            now = datetime.now()
            date_time = now.strftime("%m%d%Y_%H%M%S")

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            video_dir = os.path.join(recording_dir, 'videos', str(cameraId))
            os.makedirs(video_dir, exist_ok=True)

            cv2.imwrite(os.path.join(depth_dir, f'depth_{posture}_camId_{cameraId}_n{i:04d}_{date_time}.png'), depth_colormap)
            np.save(os.path.join(depth_dir, f'depth_{posture}_camId_{cameraId}_n{i:04d}_{date_time}.npy'), depth_data)

            rgb_file_path = os.path.join(video_dir, f'rgb_{posture}_camId_{cameraId}_n{i:04d}_{date_time}.png')
            cv2.imwrite(rgb_file_path, color_image)

            i += 1

    finally:
        pipeline.stop()

def monitor_stop_signal():
    global stop_recording
    print("Press 'q' to stop recording...")
    keyboard.wait('q')  # Wait for the 'q' key to be pressed
    stop_recording = True

def record_participant(id, posture):
    global stop_recording
    stop_recording = False  # Initialize stop_recording

    ctx = rs.context()
    cameras_serial_numbers = []

    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            cameras_serial_numbers.append(d.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")

    run_number = get_run_number(id, posture)  # Get the number of runs for the specific combination
    recording_dir = os.path.join('.', f"{id}-{posture}-{run_number}")  # Save in current working directory

    os.makedirs(recording_dir, exist_ok=True)

    # Start the thread to monitor for the stop signal
    stop_signal_thread = threading.Thread(target=monitor_stop_signal)
    stop_signal_thread.start()

    threads = []
  
    for s in cameras_serial_numbers:
        thread = camThread(s, id, posture, recording_dir)
        thread.start()
        print('Started collection with RealSense: ', s)
        threads.append(thread)

    for t in threads:
        t.join()

    # Ensure that the monitoring thread has finished
    stop_signal_thread.join()

    print("Active threads", threading.activeCount())
    print("Data collection complete.")
