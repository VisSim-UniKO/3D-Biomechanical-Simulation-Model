import tkinter as tk
from tkinter import messagebox, scrolledtext
from multiprocessing import Process
import sys
import Pose3D
import threading

# Function to run each Pose2Sim task in a separate process
def run_in_process(func, status_label, console):
    def task():
        try:
            update_status(status_label, f"{func.__name__.replace('_', ' ').capitalize()} started", "red")
            func()
            update_status(status_label, f"{func.__name__.replace('_', ' ').capitalize()} finished", "green")
        except Exception as e:
            console.insert(tk.END, f"Error: {e}\n")

    thread = threading.Thread(target=task)
    thread.start()

# Functions to run Pose2Sim tasks
def run_mk_config_files(status_label, console):
    run_in_process(Pose3D.mk_data_struct, status_label, console)

def run_pose_estimation(status_label, console):
    run_in_process(Pose3D.poseEstimation, status_label, console)

def run_synchronization(status_label, console):
    run_in_process(Pose3D.synchronization, status_label, console)

def run_triangulation(status_label, console):
    run_in_process(Pose3D.triangulation, status_label, console)

def run_filtering(status_label, console):
    run_in_process(Pose3D.filtering, status_label, console)

def run_marker_augmentation(status_label, console):
    run_in_process(Pose3D.markerAugmentation, status_label, console)

def run_rotate_coordinates(status_label, console):
    run_in_process(Pose3D.rotate_coordinates, status_label, console)

def run_kinematics(status_label, console):
    run_in_process(Pose3D.kinematics, status_label, console)

# Function to update the status label
def update_status(label, message, color):
    label.config(text=f"Status: {message}", fg=color)

# Redirect print statements to the console in the GUI
class RedirectText:
    def __init__(self, widget):
        self.widget = widget

    def write(self, string):
        self.widget.insert(tk.END, string)
        self.widget.see(tk.END)

    def flush(self):
        pass  # Not needed for this use case

# Create a simple GUI
def run_gui():
    root = tk.Tk()
    root.title("Pose2Sim Tool GUI")

    # Add buttons for each task
    status_label = tk.Label(root, text="Status: Waiting", relief=tk.SUNKEN, anchor="w", bg="lightgray", font=("Arial", 12))
    status_label.pack(fill="x", pady=10)

    tk.Button(root, text="Make Config Files", command=lambda: run_mk_config_files(status_label, console)).pack(pady=10)
    tk.Button(root, text="Pose Estimation", command=lambda: run_pose_estimation(status_label, console)).pack(pady=10)
    tk.Button(root, text="Synchronization", command=lambda: run_synchronization(status_label, console)).pack(pady=10)
    tk.Button(root, text="Triangulation", command=lambda: run_triangulation(status_label, console)).pack(pady=10)
    tk.Button(root, text="Filtering", command=lambda: run_filtering(status_label, console)).pack(pady=10)
    tk.Button(root, text="Marker Augmentation", command=lambda: run_marker_augmentation(status_label, console)).pack(pady=10)
    tk.Button(root, text="Rotate Coordinates", command=lambda: run_rotate_coordinates(status_label, console)).pack(pady=10)
    tk.Button(root, text="Kinematics", command=lambda: run_kinematics(status_label, console)).pack(pady=10)

    # Console log area
    console_label = tk.Label(root, text="Console Log", font=("Arial", 12))
    console_label.pack(pady=5)
    
    console = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, width=50)
    console.pack(pady=5)

    # Redirect sys.stdout to console Text widget
    sys.stdout = RedirectText(console)
    sys.stderr = RedirectText(console)

    root.mainloop()

# Call run_gui() to start the GUI
if __name__ == "__main__":
    run_gui()
