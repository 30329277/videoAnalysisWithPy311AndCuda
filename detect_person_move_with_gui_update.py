import tkinter as tk
from tkinter import filedialog
from tkinter import ttk  # Import ttk for themed widgets
import torch
from torchvision import models, transforms
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock
import numpy as np
import subprocess
import os
from datetime import datetime

# Load pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO dataset category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def process_frame(frame, model, transform, target_label_id, threshold=0.8):
    # 将图像转换为Tensor并移动到GPU
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(frame_tensor)  # 在GPU上进行推理

    people_boxes = []
    for pred in range(len(predictions[0]['labels'])):
        label_id = predictions[0]['labels'][pred].item()
        if label_id == target_label_id and predictions[0]['scores'][pred] > threshold:
            people_boxes.append(predictions[0]['boxes'][pred].cpu().numpy())  # 将结果移回CPU
    
    return people_boxes

def frame_generator(video_path, target_width=640, target_height=480, interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * interval)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
        
        # 跳到下一个间隔
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_per_interval - 1)

    cap.release()

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"

def estimate_remaining_time(start_time, processed_frames, total_frames, fps):
    elapsed_time = time.time() - start_time
    avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0
    remaining_frames = total_frames - processed_frames
    remaining_time = remaining_frames * avg_time_per_frame
    return remaining_time

def calculate_motion(previous_boxes, current_boxes, movement_threshold=10):
    def box_center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    if not previous_boxes or not current_boxes:
        return False

    for prev_box in previous_boxes:
        prev_center = box_center(prev_box)
        for curr_box in current_boxes:
            curr_center = box_center(curr_box)
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            if distance > movement_threshold:
                return True
    
    return False

def convert_to_mp4(video_path):
    """Converts video to MP4 format using FFmpeg with GPU acceleration if available."""
    base, ext = os.path.splitext(video_path)
    output_path = base + ".mp4"
    ffmpeg_path = "ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"  # 使用相对路径

    # 检查同名的 MP4 文件是否已经存在
    if os.path.exists(output_path):
        print(f"MP4 file {output_path} already exists. Skipping conversion.")
        return output_path

    try:
        # 使用 nvenc 进行 GPU 加速
        subprocess.run([ffmpeg_path, '-i', video_path, '-c:v', 'h264_nvenc', '-preset', 'medium', '-crf', '23', output_path, '-y'], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return None

def detect_people_in_video(video_path, model, transform, target_label_id, interval=30, num_threads=4, threshold=0.8):
    # Check if the video is MTS and convert if necessary
    if video_path.lower().endswith(".mts"):
        print("MTS file detected. Converting to MP4...")
        converted_path = convert_to_mp4(video_path)
        if converted_path:
            video_path = converted_path
        else:
            print("Failed to convert MTS file. Exiting.")
            return -1, -1, -1

    # 获取当前时间并生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    output_file = f"{video_name}_{timestamp}_output.txt"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    detected_times = 0
    active_times = 0
    lock = Lock()
    processed_frames = 0
    previous_boxes = None
    progress_lock = Lock()
    start_time = time.time()
    last_person_detected_time = 0
    person_detected_intervals = []

    with open(output_file, "w") as f:
        def worker(frame_queue):
            nonlocal processed_frames, detected_times, active_times, previous_boxes, last_person_detected_time, person_detected_intervals
            while True:
                try:
                    frame_num, frame = frame_queue.get(timeout=1)
                    current_boxes = process_frame(frame, model, transform, target_label_id, threshold)
                    
                    if current_boxes:
                        with lock:
                            if last_person_detected_time == 0:
                                last_person_detected_time = frame_num
                            detected_times += interval
                            timestamp = format_time(frame_num / fps)
                            f.write(f"Person detected at {timestamp}, duration: {interval} seconds\n")
                        
                        if previous_boxes and calculate_motion(previous_boxes, current_boxes):
                            with lock:
                                active_times += interval
                                f.write(f"Active person detected at {timestamp}, duration: {interval} seconds\n")
                        
                        previous_boxes = current_boxes
                    else:
                        if last_person_detected_time != 0:
                            with lock:
                                person_detected_intervals.append((last_person_detected_time, frame_num))
                                last_person_detected_time = 0

                    with progress_lock:
                        processed_frames += 1
                    frame_queue.task_done()
                except Queue.Empty:
                    break

        frame_queue = Queue(maxsize=num_threads * 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(worker, frame_queue)

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

        # 记录最终结果
        video_duration_formatted = format_time(video_duration)
        detected_percentage = (detected_times / video_duration) * 100 if video_duration > 0 else 0
        active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0

        f.write(f"\nTotal detected time: {detected_times} seconds ({format_time(detected_times)}), ({detected_percentage:.2f}% of video duration).\n")
        f.write(f"Total active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration).\n")

        # 输出每个检测到人的区间
        for start, end in person_detected_intervals:
            start_time_str = format_time(start / fps)
            end_time_str = format_time(end / fps)
            f.write(f"Person detected from {start_time_str} to {end_time_str}\n")

    print(f"\nProcessing frames: {processed_frames}/{total_frames} (100.00%)")
    print(f"Video analysis completed. Output written to {output_file}")

    return detected_times, active_times, fps


def browse_directory():
    directory = filedialog.askdirectory()
    if directory:
        directory_path.set(directory)
        update_video_list(directory)

def update_video_list(directory):
    for widget in inner_video_frame.winfo_children():
        widget.destroy()

    video_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.mp4', '.mts', '.avi'))]
    for i, video_file in enumerate(video_files):
        var = tk.BooleanVar()
        check = tk.Checkbutton(inner_video_frame, variable=var)
        check.grid(row=i, column=0, sticky="w")
        label = tk.Label(inner_video_frame, text=video_file)
        label.grid(row=i, column=1, sticky="w")

        # Create the result label here
        result_label = tk.Label(inner_video_frame, text="")
        result_label.grid(row=i, column=3, sticky="w")

        button = tk.Button(inner_video_frame, text="Analysis", command=lambda path=os.path.join(directory, video_file), rl=result_label: analyze_video(path, rl)) # Pass result_label to analyze_video
        button.grid(row=i, column=2, sticky="w")

        video_checkboxes.append((var, video_file, result_label)) # Store the result label

    # Update the scrollregion after adding new widgets
    inner_video_frame.update_idletasks()  # Force an update of the geometry information
    video_frame.config(scrollregion=video_frame.bbox("all"))


def analyze_video(video_path, result_label):
    detected_times, active_times, video_fps = detect_people_in_video(video_path, model, transform, target_label_id, interval=interval, num_threads=4)
    if detected_times != -1:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        detected_percentage = (detected_times / video_duration) * 100 if video_duration > 0 else 0
        active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0
        result_text = f"Total detected time: {detected_times} seconds ({format_time(detected_times)}), ({detected_percentage:.2f}% of video duration).\nTotal active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration)."
        result_label.config(text=result_text)


def mass_analysis():
    directory = directory_path.get()
    if not directory:
        mass_result.set("Please select a directory first.")
        return

    selected_videos_data = [
        (os.path.join(directory, file), result_label)
        for var, file, result_label in video_checkboxes
        if var.get() and os.path.isfile(os.path.join(directory, file))
    ]

    if not selected_videos_data:
        mass_result.set("No videos selected.")
        return

    total_detected_time = 0
    total_active_time = 0
    total_video_duration = 0
    total_videos = len(selected_videos_data)
    current_video = 0

    for video_path, result_label in selected_videos_data:
        current_video += 1
        progress_label.config(text=f"Analyzing video {current_video}/{total_videos}: {os.path.basename(video_path)}")
        root.update_idletasks()

        detected_times, active_times, video_fps = detect_people_in_video(video_path, model, transform, target_label_id, interval=interval, num_threads=4)
        if detected_times == -1:
            mass_result.set(f"Error analyzing {video_path}.")
            progress_label.config(text="Error!")
            root.update_idletasks()
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()

        # Update individual video result label
        detected_percentage = (detected_times / video_duration) * 100 if video_duration > 0 else 0
        active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0
        result_text = f"Total detected time: {detected_times} seconds ({format_time(detected_times)}), ({detected_percentage:.2f}% of video duration).\nTotal active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of total video duration)."
        result_label.config(text=result_text)


        total_detected_time += detected_times
        total_active_time += active_times
        total_video_duration += video_duration

    detected_percentage = (total_detected_time / total_video_duration) * 100 if total_video_duration > 0 else 0
    active_percentage = (total_active_time / total_video_duration) * 100 if total_video_duration > 0 else 0
    result_text = f"Total detected time (all videos): {total_detected_time} seconds ({format_time(total_detected_time)}), ({detected_percentage:.2f}% of total video duration).\nTotal active time (all videos): {total_active_time} seconds ({format_time(total_active_time)}), ({active_percentage:.2f}% of total video duration)."
    mass_result.set(result_text)
    progress_label.config(text="Finished!")
    root.update_idletasks()


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
target_label_id = 1
mass_result = tk.StringVar()

# --- Part 1: Directory selection ---
frame_directory = ttk.Frame(root)
frame_directory.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

browse_button = ttk.Button(frame_directory, text="选择目录", command=browse_directory)
browse_button.pack(side="left")

directory_label = ttk.Label(frame_directory, textvariable=directory_path, width=50)
directory_label.pack(side="left", padx=(10, 0))  # Add padding


# Function to select or deselect all videos
def select_deselect_all(select):
    for var, _, _ in video_checkboxes:
        var.set(select)

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
    # Reuse the logic from mass_analysis function here
    mass_analysis()

# - Part 2: Video list -
frame_videos = ttk.Frame(root)
frame_videos.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

# Add buttons for Select All and Deselect All
select_all_button = ttk.Button(frame_videos, text="全选", command=lambda: select_deselect_all(True))
select_all_button.pack(side="top", anchor="nw", pady=(0, 5))

deselect_all_button = ttk.Button(frame_videos, text="取消全选", command=lambda: select_deselect_all(False))
deselect_all_button.pack(side="top", anchor="nw")

# Add a button for Analyze All
# analyze_all_button = ttk.Button(frame_videos, text="全部分析", command=analyze_all_videos)
# analyze_all_button.pack(side="top", anchor="ne", pady=(0, 5))

video_frame = tk.Canvas(frame_videos) # Use a Canvas for scrolling
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
frame_analysis.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 0), sticky="ew") # Remove bottom padding

mass_analysis_button = ttk.Button(frame_analysis, text="批量分析", command=mass_analysis)
mass_analysis_button.pack(fill='x', pady=(0,5)) # Remove top padding, keep bottom padding


# Progress indicator
progress_label = ttk.Label(frame_analysis, text="Ready")
progress_label.pack(pady=(0,5)) # Remove top padding, keep small bottom padding


mass_result_label = ttk.Label(root, textvariable=mass_result, wraplength=760, justify="left")
mass_result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 0), sticky="nw")

root.grid_rowconfigure(1, weight=1)  # Video list area expands
root.grid_rowconfigure(2, minsize=60)  # Reduced minimum height for the analysis section (adjust as needed)
root.grid_rowconfigure(3, minsize=60)  # Increased minimum height for the results section
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()