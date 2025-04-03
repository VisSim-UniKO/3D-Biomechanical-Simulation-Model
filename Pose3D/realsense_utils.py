import pyrealsense2 as rs
import numpy as np

def create_pipelines(num_cams):
    """
    Create and configure RealSense pipelines for the specified number of cameras.
    Returns the pipelines and camera serial numbers.
    """
    ctx = rs.context()
    serial_numbers = []
    
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            serial_numbers.append(d.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
        exit(0)

    assert len(serial_numbers) >= num_cams, f'Required {num_cams} cameras, but found only {len(serial_numbers)}'

    pipelines = []
    camera_serial_numbers = []

    for serial in serial_numbers[:num_cams]:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        
        device = pipeline_profile.get_device()
        camera_serial_number = device.get_info(rs.camera_info.serial_number)
        camera_serial_numbers.append(camera_serial_number)

        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                break
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = pipeline.start(config)
        
        pipelines.append(pipeline)

    return pipelines, camera_serial_numbers
