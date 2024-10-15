import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock
import numpy as np

# Function to browse and select a directory
def browse_directory():
    dir_path = filedialog.askdirectory()
    if dir_path:
        directory_path.set(dir_path)
        load_videos()

# Function to load video files from the selected directory
def load_videos():
    global video_checkboxes
    video_checkboxes.clear()
    for widget in inner_video_frame.winfo_children():
        widget.destroy()

    if not directory_path.get():
        return

    for file in os.listdir(directory_path.get()):
        if file.lower().endswith(('.mp4', '.avi', '.mkv')):
            var = tk.BooleanVar(value=False)
            checkbox = ttk.Checkbutton(inner_video_frame, text=file, variable=var)
            checkbox.pack(anchor="w")
            analyze_button = ttk.Button(inner_video_frame, text="分析", command=lambda f=file: analyze_single_video(f))
            analyze_button.pack(anchor="e")
            result_label = ttk.Label(inner_video_frame, text="")
            result_label.pack(anchor="w")
            video_checkboxes.append((var, file, result_label))

# Function to analyze a single video
def analyze_single_video(video_file):
    full_path = os.path.join(directory_path.get(), video_file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(detect_motion_in_video, full_path, interval=interval)
        future.add_done_callback(lambda f: update_result(video_file, f.result()))

# Function to analyze all selected videos
def analyze_all_videos():
    selected_videos_data = [
        (os.path.join(directory_path.get(), file), result_label)
        for var, file, result_label in video_checkboxes
        if var.get() and os.path.isfile(os.path.join(directory_path.get(), file))
    ]
    if not selected_videos_data:
        mass_result.set("没有选择任何视频。")
        return

    # Clear previous results
    for _, _, result_label in video_checkboxes:
        result_label.config(text="")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(detect_motion_in_video, path, interval): (path, result_label) for path, result_label in selected_videos_data}
        for future in concurrent.futures.as_completed(futures):
            path, result_label = futures[future]
            try:
                result = future.result()
                update_result(path, result)
            except Exception as e:
                result_label.config(text=f"错误: {str(e)}")

# Function to select or deselect all videos
def select_deselect_all(select):
    for var, _, _ in video_checkboxes:
        var.set(select)

# Function to process frame using optical flow
def process_frame(frame, prev_gray, threshold=2.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return gray, False

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    _, motion_mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    has_motion = np.sum(motion_mask) > 100  # Adjust threshold as needed

    return gray, has_motion

# Frame generator
def frame_generator(video_path, target_width=320, target_height=240, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * interval)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_per_interval - 1)
    cap.release()

# Helper functions
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"

def estimate_remaining_time(start_time, processed_frames, total_frames, fps):
    elapsed_time = time.time() - start_time
    avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0
    remaining_frames = total_frames - processed_frames
    remaining_time = avg_time_per_frame * remaining_frames
    return remaining_time

# Main function to detect motion in video
def detect_motion_in_video(video_path, interval=0.5, num_threads=4, threshold=2.0):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps
    cap.release()

    active_times = 0
    lock = Lock()
    processed_frames = 0
    progress_lock = Lock()
    start_time = time.time()
    last_activity_detected_time = 0
    activity_detected_intervals = []

    output_file = "output.txt"
    with open(output_file, "w") as f:
        def worker(frame_queue, f):
            nonlocal processed_frames, active_times, last_activity_detected_time, activity_detected_intervals
            prev_gray = None
            while True:
                try:
                    frame_num, frame = frame_queue.get(timeout=1)
                    gray, has_motion = process_frame(frame, prev_gray, threshold)
                    if has_motion:
                        with lock:
                            if last_activity_detected_time == 0:
                                last_activity_detected_time = frame_num
                            active_times += interval
                            timestamp = format_time(frame_num / fps)
                            f.write(f"Activity detected at {timestamp}, duration: {interval} seconds\n")
                    else:
                        if last_activity_detected_time != 0:
                            with lock:
                                activity_detected_intervals.append((last_activity_detected_time, frame_num))
                                last_activity_detected_time = 0
                    prev_gray = gray
                    with progress_lock:
                        processed_frames += 1
                    frame_queue.task_done()
                except Queue.Empty:
                    break

        frame_queue = Queue(maxsize=num_threads * 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(worker, frame_queue, f)

            try:
                for frame_num, frame in frame_generator(video_path, interval=interval):
                    frame_queue.put((frame_num, frame))
                    with progress_lock:
                        current_progress = (processed_frames / total_frames) * 100
                        remaining_time = estimate_remaining_time(start_time, processed_frames, total_frames, fps)
                        remaining_time_formatted = format_time(remaining_time)
                        print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%), Estimated time remaining: {remaining_time_formatted}", end='\r')
            except Exception as e:
                print(f"An error occurred during processing: {e}")
                return -1, -1, -1
        frame_queue.join()

    video_duration_formatted = format_time(video_duration)
    active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0
    with open(output_file, "a") as f:
        f.write(f"\nTotal active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration).\n")

    return active_times, active_percentage, fps

# Function to update the result label in the UI
def update_result(video_path, result):
    for _, file, result_label in video_checkboxes:
        if os.path.basename(video_path) == file:
            result_label.config(text=f"分析结果: {result[0]} 秒 ({result[1]:.2f}%)")

# GUI setup
root = tk.Tk()
root.title("Video Analysis")
root.geometry("800x500")
root.resizable(False, False)

# Use ttk styles for a more modern look
style = ttk.Style(root)
style.theme_use("clam")  # Or another available theme

# Variables
directory_path = tk.StringVar()
video_checkboxes = []
interval = 10
mass_result = tk.StringVar()

# --- Part 1: Directory selection ---
frame_directory = ttk.Frame(root)
frame_directory.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

browse_button = ttk.Button(frame_directory, text="选择目录", command=browse_directory)
browse_button.pack(side="left")

directory_label = ttk.Label(frame_directory, textvariable=directory_path, width=50)
directory_label.pack(side="left", padx=(10, 0))  # Add padding

# - Part 2: Video list -
frame_videos = ttk.Frame(root)
frame_videos.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

# Add buttons for Select All and Deselect All
select_all_button = ttk.Button(frame_videos, text="全选", command=lambda: select_deselect_all(True))
select_all_button.pack(side="top", anchor="nw", pady=(0, 5))

deselect_all_button = ttk.Button(frame_videos, text="取消全选", command=lambda: select_deselect_all(False))
deselect_all_button.pack(side="top", anchor="nw")

# Add a button for Analyze All
analyze_all_button = ttk.Button(frame_videos, text="全部分析", command=analyze_all_videos)
analyze_all_button.pack(side="top", anchor="ne", pady=(0, 5))

video_frame = tk.Canvas(frame_videos)  # Use a Canvas for scrolling
video_frame.pack(side="left", fill="both", expand=True)
scrollbar = ttk.Scrollbar(frame_videos, orient="vertical", command=video_frame.yview)
scrollbar.pack(side="right", fill="y")
video_frame.configure(yscrollcommand=scrollbar.set)
video_frame.bind('<Configure>', lambda e: video_frame.configure(scrollregion=video_frame.bbox("all")))

# Define inner_video_frame as a global variable
global inner_video_frame
inner_video_frame = ttk.Frame(video_frame)
video_frame.create_window((0, 0), window=inner_video_frame, anchor="nw")

# Bind the function to the configure event of the inner frame
inner_video_frame.bind("<Configure>", lambda e: video_frame.config(scrollregion=video_frame.bbox("all")))

# --- Part 3: Mass analysis ---
frame_analysis = ttk.Frame(root)
frame_analysis.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 0), sticky="ew")  # Remove bottom padding

mass_analysis_button = ttk.Button(frame_analysis, text="批量分析", command=analyze_all_videos)
mass_analysis_button.pack(fill='x', pady=(0, 5))  # Remove top padding, keep bottom padding

# Progress indicator
progress_label = ttk.Label(frame_analysis, text="Ready")
progress_label.pack(pady=(0, 5))  # Remove top padding, keep small bottom padding

mass_result_label = ttk.Label(root, textvariable=mass_result, wraplength=760, justify="left")
mass_result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 0), sticky="nw")

root.grid_rowconfigure(1, weight=1)  # Video list area expands
root.grid_rowconfigure(2, minsize=60)  # Reduced minimum height for the analysis section (adjust as needed)
root.grid_rowconfigure(3, minsize=60)  # Increased minimum height for the results section
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()