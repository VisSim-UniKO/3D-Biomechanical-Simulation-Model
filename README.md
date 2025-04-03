# Pose3D
Framework for the [Pose2Sim](https://github.com/perfanalytics/pose2sim?tab=readme-ov-file) method to build a 3D biomechanical simulation model for OpenSim.

Including tools for Recording and Extrinsic Calibration of multiple Realsense D435i cameras.

## Installation
#### 1. Anaconda
Install anaconda or [miniconda](https://docs.anaconda.com/miniconda/).
Once installed, open an Anaconda prompt and create a virtual environment:

`conda create -n Pose3D python=3.9`

`conda activate Pose3D`

_Optional_: initialize anaconda to use directly from your command prompt:
`conda init cmd.exe` or `conda init powershell`

_Optional_: to use the Pose2Sim Tool from the GUI, you need to install Tkinker.
`conda install tk`


#### 2. Opensim
Install the OpenSIM Python API

`conda install -c opensim-org opensim -y`

#### 3. Pose2Sim and Pose2Sim Tool
Install [Pose2Sim](https://github.com/perfanalytics/pose2sim?tab=readme-ov-file) and all additional tools in this repository with:

`pip install git+ssh://git@gitlab.uni-koblenz.de/VisSim/VisSim-OpenSim/pose2sim.git`


## How to use

### Data Collection

**Start Pose3D**

In your terminal (anaconda prompt, or cmd/powershell if its initialized), navigate to your directory and activate the Pose3D environment: `conda activate Pose3D`

Start python: `python` or `ipython`

Import the PoseSim_Tool: `from Pose3D import Pose3D`

Now you can use the following tools.

**Create file structure**

Your current working directory should represent one batch session, meaning multiple recordings with the same camera setup.
You need the following data structure, which can be created with the Pose2Sim Tool:

<pre>
batch-session-directory/
├── calibration/
│   ├── intrinsics/
│   └── extrinsics/
├── Recording-A/
│   └── videos/
│       ├── cam1/
│       ├── cam2/
│       └── ...
├── Recording-B/
│   └── videos/
│       ├── cam1/
│       ├── cam2/
│       └── ...
├── ...
└── Participant_data.csv
</pre>

**Setup for calibration**


The _calibration_ directory includes the cameras intrinsics and extrinsics. 
You need to set up your cameras, then run `Pose3D.extract_intrinsics()`.

This will save the intrinsic parameters of all connected realsense cameras to a file _calibration/Calib_board.toml_.
For extrinsic calibration, you need to record a checkerboard that is visible with all cameras.
Position the checkerboard and record with `Pose3D.record_checkerboard()`.

FYI: We only need one frame per camera, so you can later choose one frame (but make sure to use ones with the same timestamp for all cams), otherwise the extrinsic calibration just uses the first frame.

**Save participant data**

TODO: add image

Every participant gets a unique _ID_ that should be used for every recording with this participant.
Their demographic data and additional information should be saved in a .csv file called "Participant_Data.csv".
You can create such a file with `Pose3D.create_participant_data()` and then add all their information.

**Record participant videos**

Run `Pose3D.record_participant(id=_, posture=_)` to record a participant and save the camera data (rgb and depth images). The first parameter _id_ is the participant ID. The second parameter _posture_ defines the posture or movement that is recorded. You can make multiple recordings with the same ID and posture.
For example, you can execute `Pose3D.record_participant(id="A", posture="squat")`.


### Run Pose2Sim

(you can find demo data for testing in the Pose3D repository. You can see where this was installed with `pip show Pose3D`)


Import: `from Pose3D import Pose3D`

Execute all steps: `Pose3D.runAll(do_calibration=True)`

-> (In the demo, the extrinsic calibration is already done, so you can skip this step and run all with `Pose3D.runAll(do_calibration=False)`)


~~OR: Execute the steps indiviually:
1. Create Configuration Files using the participant data: `Pose3D.make_config()`.
2. Execute Extrinsic calibration: `Pose3D.calibration()`, then follow the instructions.
2. Estimate Pose using the Pose model: `Pose3D.poseEstimation()`.
3. Synchronize the recorded video: `Pose3D.synchronization()`.
4. Triangulate 2D poses to 3D pose: `Pose3D.triangulation()`.
5. Filter the 3D pose movement: `Pose3D.filtering()`.
6. (skip this step when calculating the muscle model) Create markers: `Pose3D.markerAugmentation()`.
7. Rotate the marker coordinates to match the OpenSim coordinate system: `Pose3D.rotate_coordinates()`.
8. Calculate Inverse kinematics in OpenSim: `Pose3D.kinematics()`.~~


