#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## CAMERAS CALIBRATION                                                   ##
###########################################################################

Use this module to calibrate your cameras and save results to a .toml file.

It either converts a Qualisys calibration .qca.txt file,
Or calibrates cameras from checkerboard images.

Checkerboard calibration is based on 
https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.

INPUTS: 
- a calibration file in the 'calibration' folder (.qca.txt extension)
- OR folders 'calibration/intrinsics' (populated with video or about 30 images) and 'calibration/extrinsics' (populated with video or one image)
- a Config.toml file in the 'User' folder

OUTPUTS: 
- a calibration file in the 'calibration' folder (.toml extension)
'''

# TODO: DETECT WHEN WINDOW IS CLOSED
# TODO: WHEN 'Y', CATCH IF NUMBER OF IMAGE POINTS CLICKED NOT EQUAL TO NB OBJ POINTS


## INIT
from Pose2Sim.common import world_to_camera_persp, rotate_cam, quat2mat, euclidean_distance, natural_sort_key, zup2yup

import os
import logging
import pickle
import numpy as np
import pandas as pd
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2
import glob
import toml
import re
from lxml import etree
import warnings
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from PIL import Image
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"



def calib_calc_fun(calib_dir, intrinsics_config_dict, extrinsics_config_dict):
    '''
    Calibrates intrinsic and extrinsic parameters
    from images or videos of a checkerboard
    or retrieve them from a file

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)
    - extrinsics_config_dict: dictionary of extrinsics parameters (calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - ret: residual reprojection error in _px_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''
    
    overwrite_intrinsics = intrinsics_config_dict.get('overwrite_intrinsics')
    calculate_extrinsics = extrinsics_config_dict.get('calculate_extrinsics')

    print(f'\nCalibrating cameras...')
    print("Overwrite intrinsics: ", overwrite_intrinsics)
    print("Calculate extrinsics: ", calculate_extrinsics)

    # retrieve intrinsics if calib_file found and if overwrite_intrinsics=False
    try:
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]
    except:
        pass
    if not overwrite_intrinsics and 'calib_file' in locals():
        logging.info(f'\nPreexisting calibration file found: \'{calib_file}\'.')
        logging.info(f'\nRetrieving intrinsic parameters from file. Set "overwrite_intrinsics" to true in Config.toml to recalculate them.')
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]
        calib_data = toml.load(calib_file)

        ret, C, S, D, K, R, T = [], [], [], [], [], [], []
        for cam in calib_data:
            if cam != 'metadata':
                ret += [0.0]
                C += [calib_data[cam]['name']]
                S += [calib_data[cam]['size']]
                K += [np.array(calib_data[cam]['matrix'])]
                D += [calib_data[cam]['distortions']]
                R += [[0.0, 0.0, 0.0]]
                T += [[0.0, 0.0, 0.0]]
        nb_cams_intrinsics = len(C)
    
    # calculate intrinsics otherwise
    else:
        logging.info(f'\nCalculating intrinsic parameters...')
        ret, C, S, D, K, R, T = calibrate_intrinsics(calib_dir, intrinsics_config_dict)
        nb_cams_intrinsics = len(C)

    # calculate extrinsics
    if calculate_extrinsics:
        logging.info(f'\nCalculating extrinsic parameters...')
        
        # check that the number of cameras is consistent
        nb_cams_extrinsics = len(next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1])
        if nb_cams_intrinsics != nb_cams_extrinsics:
            raise Exception(f'Error: The number of cameras is not consistent:\
                    Found {nb_cams_intrinsics} cameras based on the number of intrinsic folders or on calibration file data,\
                    and {nb_cams_extrinsics} cameras based on the number of extrinsic folders.')
        ret, C, S, D, K, R, T = calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D)
    else:
        logging.info(f'\nExtrinsic parameters won\'t be calculated. Set "calculate_extrinsics" to true in Config.toml to calculate them.')

    return ret, C, S, D, K, R, T


def calibrate_intrinsics(calib_dir, intrinsics_config_dict):
    '''
    Calculate intrinsic parameters
    from images or videos of a checkerboard
    Extract frames, then detect corners, then calibrate

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)

    OUTPUTS:
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    '''

    try:
        intrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'intrinsics')))[1]
    except StopIteration:
        logging.exception(f'Error: No {os.path.join(calib_dir, "intrinsics")} folder found.')
        raise Exception(f'Error: No {os.path.join(calib_dir, "intrinsics")} folder found.')
    intrinsics_extension = intrinsics_config_dict.get('intrinsics_extension')
    extract_every_N_sec = intrinsics_config_dict.get('extract_every_N_sec')
    overwrite_extraction = False
    show_detection_intrinsics = intrinsics_config_dict.get('show_detection_intrinsics')
    intrinsics_corners_nb = intrinsics_config_dict.get('intrinsics_corners_nb')
    intrinsics_square_size = intrinsics_config_dict.get('intrinsics_square_size') / 1000 # convert to meters
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []

    for i,cam in enumerate(intrinsics_cam_listdirs_names):
        # Prepare object points
        objp = np.zeros((intrinsics_corners_nb[0]*intrinsics_corners_nb[1],3), np.float32) 
        objp[:,:2] = np.mgrid[0:intrinsics_corners_nb[0],0:intrinsics_corners_nb[1]].T.reshape(-1,2)
        objp[:,:2] = objp[:,0:2]*intrinsics_square_size
        objpoints = [] # 3d points in world space
        imgpoints = [] # 2d points in image plane

        logging.info(f'\nCamera {cam}:')
        img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.{intrinsics_extension}'))
        if len(img_vid_files) == 0:
            logging.exception(f'The folder {os.path.join(calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
            raise ValueError(f'The folder {os.path.join(calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
        img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers
        
        # extract frames from video if video
        try:
            cap = cv2.VideoCapture(img_vid_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            extract_frames(img_vid_files[0], extract_every_N_sec, overwrite_extraction)
            img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.png'))
            img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)])
        except:
            pass

        # find corners
        for img_path in img_vid_files:
            if show_detection_intrinsics == True:
                imgp_confirmed, objp_confirmed = findCorners(img_path, intrinsics_corners_nb, objp=objp, show=show_detection_intrinsics)
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp_confirmed)
            else:
                imgp_confirmed = findCorners(img_path, intrinsics_corners_nb, objp=objp, show=show_detection_intrinsics)
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp)
        if len(imgpoints) < 10:
            logging.info(f'Corners were detected only on {len(imgpoints)} images for camera {cam}. Calibration of intrinsic parameters may not be accurate with fewer than 10 good images of the board.')

        # calculate intrinsics
        img = cv2.imread(str(img_path))
        objpoints = np.array(objpoints)
        ret_cam, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], 
                                    None, None, flags=(cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT))
        h, w = [np.float32(i) for i in img.shape[:-1]]
        ret.append(ret_cam)
        C.append(cam)
        S.append([w, h])
        D.append(dist[0])
        K.append(mtx)
        R.append([0.0, 0.0, 0.0])
        T.append([0.0, 0.0, 0.0])
        
        logging.info(f'Intrinsics error: {np.around(ret_cam, decimals=3)} px for each cameras.')

    return ret, C, S, D, K, R, T


def calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D):
    '''
    Calibrates extrinsic parameters
    from an image or the first frame of a video
    of a checkerboard or of measured clues on the scene

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - extrinsics_config_dict: dictionary of extrinsics parameters (extrinsics_method, calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''

    try:
        extrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1]
    except StopIteration:
        logging.exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')
        raise Exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')
    
    extrinsics_method = extrinsics_config_dict.get('extrinsics_method')
    ret, R, T = [], [], []
    
    if extrinsics_method in ('board', 'scene'):
    
                
        # Define 3D object points
        if extrinsics_method == 'board':
            extrinsics_corners_nb = extrinsics_config_dict.get('board').get('extrinsics_corners_nb')
            extrinsics_square_size = extrinsics_config_dict.get('board').get('extrinsics_square_size') / 1000 # convert to meters
            object_coords_3d = np.zeros((extrinsics_corners_nb[0] * extrinsics_corners_nb[1], 3), np.float32)
            object_coords_3d[:, :2] = np.mgrid[0:extrinsics_corners_nb[0], 0:extrinsics_corners_nb[1]].T.reshape(-1, 2)
            object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * extrinsics_square_size
        elif extrinsics_method == 'scene':
            object_coords_3d = np.array(extrinsics_config_dict.get('scene').get('object_coords_3d'), np.float32)
                
        # Save reference 3D coordinates as trc
        calib_output_path = os.path.join(calib_dir, f'Object_points.trc')
        trc_write(object_coords_3d, calib_output_path)
    
        for i, cam in enumerate(extrinsics_cam_listdirs_names):
            logging.info(f'\nCamera {cam}:')
            
            # Read images or video
            extrinsics_extension = [extrinsics_config_dict.get('board').get('extrinsics_extension') if extrinsics_method == 'board'
                                    else extrinsics_config_dict.get('scene').get('extrinsics_extension')][0]
            show_reprojection_error = [extrinsics_config_dict.get('board').get('show_reprojection_error') if extrinsics_method == 'board'
                                    else extrinsics_config_dict.get('scene').get('show_reprojection_error')][0]
            img_vid_files = glob.glob(os.path.join(calib_dir, 'extrinsics', cam, f'*.{extrinsics_extension}'))
            if len(img_vid_files) == 0:
                logging.exception(f'The folder {os.path.join(calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
                raise ValueError(f'The folder {os.path.join(calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
            img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers
            
            # extract frames from image, or from video if imread is None
            img = cv2.imread(img_vid_files[0])
            if img is None:
                cap = cv2.VideoCapture(img_vid_files[0])
                res, img = cap.read()
                if res == False:
                    raise
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Find corners or label by hand
            if extrinsics_method == 'board':
                imgp, objp = findCorners(img_vid_files[0], extrinsics_corners_nb, objp=object_coords_3d, show=show_reprojection_error)
                if len(imgp) == 0:
                    logging.exception('No corners found. Set "show_detection_extrinsics" to true to click corners by hand, or change extrinsic_board_type to "scene"')
                    raise ValueError('No corners found. Set "show_detection_extrinsics" to true to click corners by hand, or change extrinsic_board_type to "scene"')

            elif extrinsics_method == 'scene':
                imgp, objp = imgp_objp_visualizer_clicker(img, imgp=[], objp=object_coords_3d, img_path=img_vid_files[0])
                if len(imgp) == 0:
                    logging.exception('No points clicked (or fewer than 6). Press \'C\' when the image is displayed, and then click on the image points corresponding to the \'object_coords_3d\' you measured and wrote down in the Config.toml file.')
                    raise ValueError('No points clicked (or fewer than 6). Press \'C\' when the image is displayed, and then click on the image points corresponding to the \'object_coords_3d\' you measured and wrote down in the Config.toml file.')
                if len(objp) < 10:
                    logging.info(f'Only {len(objp)} reference points for camera {cam}. Calibration of extrinsic parameters may not be accurate with fewer than 10 reference points, as spread out in the captured volume as possible.')
            
            elif extrinsics_method == 'keypoints':
                logging.info('Calibration based on keypoints is not available yet.')
            
            # Calculate extrinsics
            mtx, dist = np.array(K[i]), np.array(D[i])
            _, r, t = cv2.solvePnP(np.array(objp), imgp, mtx, dist)
            r, t = r.flatten(), t.flatten()

            # Projection of object points to image plane
            # # Former way, distortions used to be ignored
            # Kh_cam = np.block([mtx, np.zeros(3).reshape(3,1)])
            # r_mat, _ = cv2.Rodrigues(r)
            # H_cam = np.block([[r_mat,t.reshape(3,1)], [np.zeros(3), 1 ]])
            # P_cam = Kh_cam @ H_cam
            # proj_obj = [ ( P_cam[0] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)),  P_cam[1] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)) ) for o in objp]
            proj_obj = np.squeeze(cv2.projectPoints(objp,r,t,mtx,dist)[0])

            # Check calibration results
            if show_reprojection_error:
                # Reopen image, otherwise 2 sets of text are overlaid
                img = cv2.imread(img_vid_files[0])
                if img is None:
                    cap = cv2.VideoCapture(img_vid_files[0])
                    res, img = cap.read()
                    if res == False:
                        raise
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for o in proj_obj:
                    cv2.circle(img, (int(o[0]), int(o[1])), 8, (0,0,255), -1) 
                for i in imgp:
                    cv2.drawMarker(img, (int(i[0][0]), int(i[0][1])), (0,255,0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(img, 'Verify calibration results, then close window.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
                cv2.putText(img, 'Verify calibration results, then close window.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA) 
                cv2.drawMarker(img, (20,40), (0,255,0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(img, '    Clicked points', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
                cv2.putText(img, '    Clicked points', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
                cv2.circle(img, (20,60), 8, (0,0,255), -1)    
                cv2.putText(img, '    Reprojected object points', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
                cv2.putText(img, '    Reprojected object points', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
                im_pil = Image.fromarray(img)
                im_pil.show(title = os.path.basename(img_vid_files[0]))

            # Calculate reprojection error
            imgp_to_objreproj_dist = [euclidean_distance(proj_obj[n], imgp[n]) for n in range(len(proj_obj))]
            rms_px = np.sqrt(np.sum([d**2 for d in imgp_to_objreproj_dist]))
            ret.append(rms_px)
            R.append(r)
            T.append(t)
        
    elif extrinsics_method == 'keypoints':
        raise NotImplementedError('This has not been integrated yet.')
    
    else:
        raise ValueError('Wrong value for extrinsics_method')

    return ret, C, S, D, K, R, T


def findCorners(img_path, corner_nb, objp=[], show=True):
    '''
    Find corners in the photo of a checkerboard.
    Press 'Y' to accept detection, 'N' to dismiss this image, 'C' to click points by hand.
    Left click to add a point, right click to remove the last point.
    Use mouse wheel to zoom in and out and to pan.
    
    Make sure that: 
    - the checkerboard is surrounded by a white border
    - rows != lines, and row is even if lines is odd (or conversely)
    - it is flat and without reflections
    - corner_nb correspond to _internal_ corners
    
    INPUTS:
    - img_path: path to image (or video)
    - corner_nb: [H, W] internal corners in checkerboard: list of two integers [4,7]
    - optionnal: show: choose whether to show corner detections
    - optionnal: objp: array [3d corner coordinates]

    OUTPUTS:
    - imgp_confirmed: array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    
    img = cv2.imread(img_path)
    if img is None:
        cap = cv2.VideoCapture(img_path)
        ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find corners
    ret, corners = cv2.findChessboardCornersSB(gray, corner_nb, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    # If corners are found, refine corners
    if ret == True: 
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        logging.info(f'{os.path.basename(img_path)}: Corners found.')
        
        if show:
            # Draw corners
            cv2.drawChessboardCorners(img, corner_nb, imgp, ret)
            # Add corner index 
            for i, corner in enumerate(imgp):
                if i in [0, corner_nb[0]-1, corner_nb[0]*(corner_nb[1]-1), corner_nb[0]*corner_nb[1] -1]:
                    x, y = corner.ravel()
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 7) 
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2) 
            
            # Visualizer and key press event handler
            for var_to_delete in ['imgp_confirmed', 'objp_confirmed']:
                if var_to_delete in globals():
                    del globals()[var_to_delete]
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=imgp, objp=objp, img_path=img_path)
        else:
            imgp_objp_confirmed = imgp

    # If corners are not found, dismiss or click points by hand
    else:
        logging.info(f'{os.path.basename(img_path)}: Corners not found. To label them by hand, set "show_detection_intrinsics" to true in the Config.toml file.')
        if show:
            # Visualizer and key press event handler
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=[], objp=objp, img_path=img_path)
        else:
            imgp_objp_confirmed = []

    return imgp_objp_confirmed


def imgp_objp_visualizer_clicker(img, imgp=[], objp=[], img_path=''):
    '''
    Shows image img. 
    If imgp is given, displays them in green
    If objp is given, can be displayed in a 3D plot if 'C' is pressed.
    If img_path is given, just uses it to name the window

    If 'Y' is pressed, closes all and returns confirmed imgp and (if given) objp
    If 'N' is pressed, closes all and returns nothing
    If 'C' is pressed, allows clicking imgp by hand. If objp is given:
        Displays them in 3D as a helper. 
        Left click to add a point, right click to remove the last point.
        Press 'H' to indicate that one of the objp is not visible on image
        Closes all and returns imgp and objp if all points have been clicked
    Allows for zooming and panning with middle click
    
    INPUTS:
    - img: image opened with openCV
    - optional: imgp: detected image points, to be accepted or not. Array of [[2d corner coordinates]]
    - optionnal: objp: array of [3d corner coordinates]
    - optional: img_path: path to image

    OUTPUTS:
    - imgp_confirmed: image points that have been correctly identified. array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''
    global old_image_path
    old_image_path = img_path
                                 
    def on_key(event):
        '''
        Handles key press events:
        'Y' to return imgp, 'N' to dismiss image, 'C' to click points by hand.
        Left click to add a point, 'H' to indicate it is not visible, right click to remove the last point.
        '''

        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count
        
        if event.key == 'y':
            # If 'y', close all
            # If points have been clicked, imgp_confirmed is returned, else imgp
            # If objp is given, objp_confirmed is returned in addition
            if 'scat' not in globals() or 'imgp_confirmed' not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = np.array([imgp.astype('float32') for imgp in imgp_confirmed])
                objp_confirmed = objp_confirmed
            # OpenCV needs at leas 4 correspondance points to calibrate
            if len(imgp_confirmed) < 6:
                objp_confirmed = []
                imgp_confirmed = []
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close('all')
            if len(objp) == 0:
                if 'objp_confirmed' in globals():
                    del objp_confirmed

        if event.key == 'n' or event.key == 'q':
            # If 'n', close all and return nothing
            plt.close('all')
            imgp_confirmed = []
            objp_confirmed = []

        if event.key == 'c':
            # TODO: RIGHT NOW, IF 'C' IS PRESSED ANOTHER TIME, OBJP_CONFIRMED AND IMGP_CONFIRMED ARE RESET TO []
            # We should reopen a figure without point on it
            img_for_pointing = cv2.imread(old_image_path)
            if img_for_pointing is None:
                cap = cv2.VideoCapture(old_image_path)
                ret, img_for_pointing = cap.read()
            img_for_pointing = cv2.cvtColor(img_for_pointing, cv2.COLOR_BGR2RGB)
            ax.imshow(img_for_pointing)
            # To update the image
            plt.draw()

            if 'objp_confirmed' in globals():
                del objp_confirmed
            # If 'c', allows retrieving imgp_confirmed by clicking them on the image
            scat = ax.scatter([],[],s=100,marker='+',color='g')
            plt.connect('button_press_event', on_click)
            # If objp is given, display 3D object points in black
            if len(objp) != 0 and not plt.fignum_exists(2):
                fig_3d = plt.figure()
                fig_3d.tight_layout()
                fig_3d.canvas.manager.set_window_title('Object points to be clicked')
                ax_3d = fig_3d.add_subplot(projection='3d')
                plt.rc('xtick', labelsize=5)
                plt.rc('ytick', labelsize=5)
                for i, (xs,ys,zs) in enumerate(np.float32(objp)):
                    ax_3d.scatter(xs,ys,zs, marker='.', color='k')
                    ax_3d.text(xs,ys,zs,  f'{str(i+1)}', size=10, zorder=1, color='k') 
                set_axes_equal(ax_3d)
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                if np.all(objp[:,2] == 0):
                    ax_3d.view_init(elev=-90, azim=0)
                fig_3d.show()

        if event.key == 'h':
            # If 'h', indicates that one of the objp is not visible on image
            # Displays it in red on 3D plot
            if len(objp) != 0  and 'ax_3d' in globals():
                count = [0 if 'count' not in globals() else count+1][0]
                if 'events' not in globals():
                    # retrieve first objp_confirmed_notok and plot 3D
                    events = [event]
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # if all objp have been clicked or indicated as not visible, close all
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])[:-1]
                    imgp_confirmed = np.array(np.expand_dims(scat.get_offsets(), axis=1), np.float32) 
                    plt.close('all')
                    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve other objp_confirmed_notok and plot 3D
                    events.append(event)
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
            else:
                pass


    def on_click(event):
        '''
        Detect click position on image
        If right click, last point is removed
        '''
        
        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count, xydata
        
        # Left click: Add clicked point to imgp_confirmed
        # Display it on image and on 3D plot
        if event.button == 1: 
            # To remember the event to cancel after right click
            if 'events' in globals():
                events.append(event)
            else:
                events = [event]

            # Add clicked point to image
            xydata = scat.get_offsets()
            new_xydata = np.concatenate((xydata,[[event.xdata,event.ydata]]))
            scat.set_offsets(new_xydata)
            imgp_confirmed = np.expand_dims(scat.get_offsets(), axis=1)    
            plt.draw()

            # Add clicked point to 3D object points if given
            if len(objp) != 0:
                count = [0 if 'count' not in globals() else count+1][0]
                if count==0:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [objp[count]]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # close all
                    plt.close('all')
                    # retrieve objp_confirmed
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])
                    imgp_confirmed = np.array(imgp_confirmed, np.float32)
                    # delete all
                    for var_to_delete in ['events', 'count', 'scat', 'scat_3d', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                

        # Right click: 
        # If last event was left click, remove last point and if objp given, from objp_confirmed
        # If last event was 'H' and objp given, remove last point from objp_confirmed_notok
        elif event.button == 3: # right click
            if 'events' in globals():
                # If last event was left click: 
                if 'button' in dir(events[-1]):
                    if events[-1].button == 1: 
                        # Remove lastpoint from image
                        new_xydata = scat.get_offsets()[:-1]
                        scat.set_offsets(new_xydata)
                        plt.draw()
                        # Remove last point from imgp_confirmed
                        imgp_confirmed = imgp_confirmed[:-1]
                        if len(objp) != 0:
                            if count >= 0: 
                                count -= 1
                            # Remove last point from objp_confirmed
                            objp_confirmed = objp_confirmed[:-1]
                            # remove from plot 
                            if len(ax_3d.collections) > len(objp):
                                ax_3d.collections[-1].remove()
                                fig_3d.canvas.draw()
                            
                # If last event was 'h' key
                elif events[-1].key == 'h':
                    if len(objp) != 0:
                        if count >= 1: count -= 1
                        # Remove last point from objp_confirmed_notok
                        objp_confirmed_notok = objp_confirmed_notok[:-1]
                        # remove from plot  
                        if len(ax_3d.collections) > len(objp):
                            ax_3d.collections[-1].remove()
                            fig_3d.canvas.draw()                
    

    def set_axes_equal(ax):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        From https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Write instructions
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    
    # Put image in a matplotlib figure for more controls
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(os.path.basename(img_path))
    ax.axis("off")
    for corner in imgp:
        x, y = corner.ravel()
        cv2.drawMarker(img, (int(x),int(y)), (128,128,128), cv2.MARKER_CROSS, 10, 2)
    ax.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    
    # Allow for zoom and pan in image
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    # Handles key presses to Accept, dismiss, or click points by hand
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.draw()
    plt.show(block=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams['toolbar'] = 'toolmanager'

    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
        if var_to_delete in globals():
            del globals()[var_to_delete]

    if 'imgp_confirmed' in globals() and 'objp_confirmed' in globals():
        return imgp_confirmed, objp_confirmed
    elif 'imgp_confirmed' in globals() and not 'objp_confirmed' in globals():
        return imgp_confirmed
    else:
        return


def extract_frames(video_path, extract_every_N_sec=1, overwrite_extraction=False):
    '''
    Extract frames from video 
    if has not been done yet or if overwrite==True
    
    INPUT:
    - video_path: path to video whose frames need to be extracted
    - extract_every_N_sec: extract one frame every N seconds (can be <1)
    - overwrite_extraction: if True, overwrite even if frames have already been extracted
    
    OUTPUT:
    - extracted frames in folder
    '''
    
    if not os.path.exists(os.path.splitext(video_path)[0] + '_00000.png') or overwrite_extraction:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            frame_nb = 0
            logging.info(f'Extracting frames...')
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    if frame_nb % (fps*extract_every_N_sec) == 0:
                        img_path = (os.path.splitext(video_path)[0] + '_' +str(frame_nb).zfill(5)+'.png')
                        cv2.imwrite(str(img_path), frame)
                    frame_nb+=1
                else:
                    break


def trc_write(object_coords_3d, trc_path):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - object_coords_3d: list of 3D point lists
    - trc_path: output path of the trc file

    OUTPUT:
    - trc file with 2 frames of the same 3D points
    '''

    #Header
    DataRate = CameraRate = OrigDataRate = 1
    NumFrames = 2
    NumMarkers = len(object_coords_3d)
    keypoints_names = np.arange(NumMarkers)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + os.path.basename(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(str(k) for k in keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(NumMarkers)])]
    
    # Zup to Yup coordinate system
    object_coords_3d = pd.DataFrame([np.array(object_coords_3d).flatten(), np.array(object_coords_3d).flatten()])
    object_coords_3d = zup2yup(object_coords_3d)
    
    #Add Frame# and Time columns
    object_coords_3d.index = np.array(range(0, NumFrames)) + 1
    object_coords_3d.insert(0, 't', object_coords_3d.index / DataRate)

    #Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        object_coords_3d.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def toml_write(calib_path, C, S, D, K, R, T):
    '''
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    OUTPUTS:
    - a .toml file cameras calibrations
    '''

    with open(os.path.join(calib_path), 'w+') as cal_f:
        for c in range(len(C)):
            cam=f'[{C[c]}]\n'
            name = f'name = "{C[c]}"\n'
            size = f'size = [ {S[c][0]}, {S[c][1]}]\n' 
            mat = f'matrix = [ [ {K[c][0,0]}, 0.0, {K[c][0,2]}], [ 0.0, {K[c][1,1]}, {K[c][1,2]}], [ 0.0, 0.0, 1.0]]\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]}]\n' 
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]}]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]}]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def recap_calibrate(ret, calib_path, calib_full_type):
    '''
    Print a log message giving calibration results. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''
    
    calib = toml.load(calib_path)
    
    ret_m, ret_px = [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            f_px = calib[cam]['matrix'][0][0]
            Dm = euclidean_distance(calib[cam]['translation'], [0,0,0])
            if calib_full_type in ['convert_qualisys', 'convert_vicon','convert_opencap', 'convert_biocv']:
                ret_m.append( np.around(ret[c], decimals=3) )
                ret_px.append( np.around(ret[c] / (Dm*1000) * f_px, decimals=3) )
            elif calib_full_type=='calculate':
                ret_px.append( np.around(ret[c], decimals=3) )
                ret_m.append( np.around(ret[c]*Dm*1000 / f_px, decimals=3) )
                
    logging.info(f'\n--> Residual (RMS) calibration errors for each camera are respectively {ret_px} px, \nwhich corresponds to {ret_m} mm.\n')
    logging.info(f'Calibration file is stored at {calib_path}.')


def calibrate_cams_all(config_dict):
    '''
    Either converts a preexisting calibration file, 
    or calculates calibration from scratch (from a board or from points).
    Stores calibration in a .toml file
    Prints recap.
    
    INPUTS:
    - a config_dict dictionary

    OUTPUT:
    - a .toml camera calibration file
    '''

    print("Config_dict: ", config_dict)
    

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    print(f'\n--> Calibrating cameras from {project_dir}')
    print("Possible calibs: ", [os.path.join(project_dir, c) for c in os.listdir(project_dir) if ('Calib' in c or 'calib' in c)])
    calib_dir = [os.path.join(project_dir, c) for c in os.listdir(project_dir) if ('Calib' in c or 'calib' in c)][0]
    print(f'\n--> Calibration files are stored in {calib_dir}')
    calib_type = config_dict.get('calibration').get('calibration_type')

    if calib_type=='convert':
        convert_filetype = config_dict.get('calibration').get('convert').get('convert_from')
        try:
            if convert_filetype=='qualisys':
                convert_ext = '.qca.txt'
                file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}*'))[0]
                binning_factor = config_dict.get('calibration').get('convert').get('qualisys').get('binning_factor')
            elif convert_filetype=='optitrack':
                file_to_convert_path = ['']
                binning_factor = 1
            elif convert_filetype=='vicon':
                convert_ext = '.xcp'
                file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}'))[0]
                binning_factor = 1
            elif convert_filetype=='opencap': # all files with .pickle extension
                convert_ext = '.pickle'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype=='easymocap': #intri.yml and intri.yml
                convert_ext = '.yml'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype=='biocv': # all files without extension -> now with .calib extension
                # convert_ext = 'no'
                # list_dir = os.listdir(calib_dir)
                # list_dir_noext = sorted([os.path.splitext(f)[0] for f in list_dir if os.path.splitext(f)[1]==''])
                # file_to_convert_path = [os.path.join(calib_dir,f) for f in list_dir_noext if os.path.isfile(os.path.join(calib_dir, f))]
                convert_ext = '.calib'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype=='anipose' or convert_filetype=='freemocap' or convert_filetype=='caliscope': # no conversion needed, skips this stage
                logging.info(f'\n--> No conversion needed from Caliscope, AniPose, nor from FreeMocap. Calibration skipped.\n')
                return
            else:
                convert_ext = '???'
                file_to_convert_path = ['']
                raise NameError(f'Calibration conversion from {convert_filetype} is not supported.') from None
            assert file_to_convert_path!=[]
        except:
            raise NameError(f'No file with {convert_ext} extension found in {calib_dir}.')
        
        calib_output_path = os.path.join(calib_dir, f'Calib_{convert_filetype}.toml')
        calib_full_type = '_'.join([calib_type, convert_filetype])
        args_calib_fun = [file_to_convert_path, binning_factor]
        
    elif calib_type=='calculate':
        intrinsics_config_dict = config_dict.get('calibration').get('calculate').get('intrinsics')
        extrinsics_config_dict = config_dict.get('calibration').get('calculate').get('extrinsics')
        extrinsics_method = config_dict.get('calibration').get('calculate').get('extrinsics').get('extrinsics_method')

        calib_output_path = os.path.join(calib_dir, f'Calib_{extrinsics_method}.toml')
        calib_full_type = calib_type
        args_calib_fun = [calib_dir, intrinsics_config_dict, extrinsics_config_dict]

    else:
        logging.info('Wrong calibration_type in Config.toml')
    
    # Map calib function
    calib_mapping = {
        'calculate': calib_calc_fun,
        }
    calib_fun = calib_mapping[calib_full_type]

    # Calibrate
    ret, C, S, D, K, R, T = calib_fun(*args_calib_fun)

    # Write calibration file
    toml_write(calib_output_path, C, S, D, K, R, T)
    
    # Recap message
    recap_calibrate(ret, calib_output_path, calib_full_type)
