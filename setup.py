from setuptools import setup, find_packages

setup(
    name='Pose3D',  # The name of your package
    version='0.1',
    packages=find_packages(),  # Automatically find all packages
    install_requires=[  # Dependencies installed via pip
        'pose2sim',
        'pandas>=1.5.0',
        'numpy>=1.22.4',
        'openpyxl',
        'argparse',
        'pyrealsense2',
        'opencv-python',
        'keyboard',
    ],
    package_data={
        'Pose3D': ['Setup_data/*', 'OpenSim_Setup/*', 'Demo_Batch_Squats/**/*'],  # Include Setup_data and demo files
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pose3D-gui=Pose3D.gui:run_gui',
            'pose3D-simplegui=Pose3D.simple_gui:create_file_explorer',
        ],
    },
    author='Lara Blomenkamp',
    author_email='larablomenkamp@uni-koblenz.de',
    description='Custom Pose2Sim tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
