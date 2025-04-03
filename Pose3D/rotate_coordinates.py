import os
import pandas as pd
import argparse
from Pose2Sim.Utilities import trc_Zup_to_Yup
import numpy as np
import toml
import cv2


def rotate_coordinates():
    # Get the setup data directory and participant data file paths
    cwd = os.getcwd()
    excel_file_path = os.path.join(cwd, 'Participant_Data.xlsx')

    # Load the participant data from the Excel file
    participant_data = pd.read_excel(excel_file_path)

    # Load camera calibration parameters
    camera_calibration_file = os.path.join(cwd, "calibration/Calib_board.toml")
    camera_calibration = toml.load(camera_calibration_file)
    
    # Select the first camera in the TOML file
    first_camera = next(iter(camera_calibration.keys() - {"metadata"}))
    cam_params = camera_calibration[first_camera]
    R = np.array(cam_params['rotation'])
    T = np.array(cam_params['translation'])

    print("Rotation matrix: ", R)
    print("Translation vector: ", T)



    def apply_camera_transformation(x, y, z, rotation_vector, translation_vector):
        """
        Apply camera transformation to the coordinates using Rodrigues rotation vector and translation vector.
        
        :param x: X coordinate of the point in world coordinates
        :param y: Y coordinate of the point in world coordinates
        :param z: Z coordinate of the point in world coordinates
        :param rotation_vector: Rotation vector (from solvePnP)
        :param translation_vector: Translation vector (from solvePnP)
        :return: Transformed coordinates (X, Y, Z) in the camera coordinate system
        """
        # Convert the Rodrigues rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # # Inverse of the rotation matrix is the transpose of the rotation matrix
        # rotation_matrix_inv = rotation_matrix.T
        
        # # Inverse of the translation vector
        # translation_vector_inv = -np.dot(rotation_matrix_inv, translation_vector)
        
        # Create a 3x1 vector for the coordinates (CCS)
        CCS_coord = np.array([x, y, z])
        
        # # Apply the inverse transformation (from CCS to WCS)
        # WCS_coord = np.dot(rotation_matrix_inv, CCS_coord - translation_vector_inv)

        WCS_coord = np.dot(rotation_matrix, CCS_coord - translation_vector)

        
        return WCS_coord[0], WCS_coord[1], WCS_coord[2]  # Return the transformed X, Y, Z
    
    def apply_transformation(x, y, z, rotation_matrix, translation_vector):
        
        # Create a 3x1 vector for the coordinates (CCS)
        CCS_coord = np.array([x, y, z])
        
        # # Apply the inverse transformation (from CCS to WCS)
        # WCS_coord = np.dot(rotation_matrix_inv, CCS_coord - translation_vector_inv)

        WCS_coord = np.dot(rotation_matrix, CCS_coord - translation_vector)

        
        return WCS_coord[0], WCS_coord[1], WCS_coord[2]  # Return the transformed X, Y, Z



    # Process data for every participant
    for _, row in participant_data.iterrows():
        participant_id = str(row['Participant ID'])

        # Find participant directories starting with the participant ID
        participant_dirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d)) and d.startswith(participant_id + "-")]

        if not participant_dirs:
            print(f"No participant directories found for ID {participant_id}. Skipping.")
            continue

        for participant_dir in participant_dirs:
            participant_dir_path = os.path.join(cwd, participant_dir)
            print(f"Processing Participant {participant_id} in directory {participant_dir_path}...")

            try:
                # Get trc file
                pose_path = os.path.join(participant_dir_path, 'pose-3d')
                trc_file = [f for f in os.listdir(pose_path) if f.endswith('butterworth.trc')][0]
                trc_file_path = os.path.join(pose_path, trc_file)

                # Header
                with open(trc_file_path, 'r') as trc_file:
                    header = [next(trc_file) for _ in range(5)]

                # Data
                trc_df = pd.read_csv(trc_file_path, sep="\t", skiprows=4)
                frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
                Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

                # Ensure no extra column is included
                num_columns = len(Q_coord.columns)
                if num_columns % 3 == 1:  # If there's an extra column
                    num_columns -= 1  # Remove the extra column

                
                # get the first frame
                first_frame = Q_coord.iloc[0]
                # get the x y z coordinates of the row "RAnkle" and "LAnkle"
                x_RAnkle = first_frame[9]
                y_RAnkle = first_frame[10]
                z_RAnkle = first_frame[11]

                x_LAnkle = first_frame[27]
                y_LAnkle = first_frame[28]
                z_LAnkle = first_frame[29]

                print(f"  RAnkle - X: {x_RAnkle}, Y: {y_RAnkle}, Z: {z_RAnkle}")
                print(f"  LAnkle - X: {x_LAnkle}, Y: {y_LAnkle}, Z: {z_LAnkle}")

                vector_RAnkle_LAnkle = np.array([x_LAnkle - x_RAnkle, y_LAnkle - y_RAnkle, z_LAnkle - z_RAnkle])
    
                # Normalize the vector from RAnkle to LAnkle (X-axis of LCS)
                vector_RAnkle_LAnkle_unit = vector_RAnkle_LAnkle / np.linalg.norm(vector_RAnkle_LAnkle)
                
                # Up vector (Z-axis of LCS)
                vector_up = np.array([0, 1, 0])
                
                # Cross product of the RAnkle-LAnkle vector and up vector to find the Y-axis of LCS
                vector_cross = np.cross(vector_RAnkle_LAnkle, vector_up)
                vector_cross_unit = vector_cross / np.linalg.norm(vector_cross)
                
                # Ensure orthogonality (cross product of X and Y should give Z)
                vector_z_unit = np.cross(vector_RAnkle_LAnkle_unit, vector_cross_unit)
                
                # Middle point between RAnkle and LAnkle (origin of LCS)
                middle_point = np.array([(x_RAnkle + x_LAnkle) / 2, (y_RAnkle + y_LAnkle) / 2, (z_RAnkle + z_LAnkle) / 2])
                
                # Define the rotation matrix (3x3 matrix of unit vectors)
                rotation_matrix = np.vstack([vector_RAnkle_LAnkle_unit, vector_cross_unit, vector_z_unit]).T
                
                # Define the translation vector
                translation_vector = middle_point






                #print("Qcoord before transformation:")
                #Print coordinates for each marker and frame, applying the transformation
                for frame_idx, row in Q_coord.iterrows():
                    #print(f"Frame {frame_idx + 1}:"




                    for i in range(0, num_columns, 3):  # Iterate over the columns in triplets (X, Y, Z)
                        # Get the X, Y, Z coordinates
                        x = row[i]
                        y = row[i + 1]
                        z = row[i + 2]

                        #print(f"  Marker {i//3 + 1} - X: {x}, Y: {y}, Z: {z}")

                        # If the last coordinate in the set is empty or NaN (an extra entry), ignore it
                        if i + 3 > len(row):  # If it's the last set of coordinates (i.e., no valid Z)
                            continue


                        # Apply the transformation
                        x_ccs, y_ccs, z_ccs = x, y, z = apply_transformation(x, y, z, np.linalg.inv(rotation_matrix), translation_vector)

                        row[i] = x_ccs
                        row[i + 1] = y_ccs
                        row[i + 2] = z_ccs

                        #print(f"  Marker {i//3 + 1} - X: {x_ccs}, Y: {y_ccs}, Z: {z_ccs}")

                # # Transform Q_coord from WCS into CCS
                cols = list(Q_coord.columns)
                # print("cols :")
                # print(cols)

                # Transform: Y->Z, Z->Y, X->-X
                # Reorder and invert X (first column)
                # cols = np.array([[cols[i*3], cols[i*3+2], cols[i*3+1]] for i in range(int(len(cols)/3))]).flatten()
                Q_Yup = Q_coord[cols].copy()  # Use .copy() to avoid SettingWithCopyWarning

                
                # Invert the x-axis (assuming x coordinates are in the first column)
                # Q_Yup.iloc[:, 0::3] = -Q_Yup.iloc[:, 0::3]  # Invert every first coordinate in the triplets

                # Write back to the original TRC file
                with open(trc_file_path, 'w') as trc_o:
                    [trc_o.write(line) for line in header]
                    Q_Yup.insert(0, 'Frame#', frames_col)
                    Q_Yup.insert(1, 'Time', time_col)
                    Q_Yup.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
                continue
