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

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def process_frame(frame, model, transform, threshold=0.5):
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(frame_tensor)
    boxes = []
    for pred in range(len(predictions[0]['labels'])):
        score = predictions[0]['scores'][pred].item()
        if score > threshold:
            boxes.append(predictions[0]['boxes'][pred].cpu().numpy())
    return boxes

def calculate_motion(boxes1, boxes2, iou_threshold=0.5):
    def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    for box1 in boxes1:
        for box2 in boxes2:
            if bb_intersection_over_union(box1, box2) >= iou_threshold:
                return True
    return False

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
    remaining_time = avg_time_per_frame * remaining_frames
    return remaining_time

def detect_objects_in_video(video_path, model, transform, interval=0.5, num_threads=4, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps
    cap.release()
    active_times = 0
    lock = Lock()
    processed_frames = 0
    previous_boxes = None
    progress_lock = Lock()
    start_time = time.time()
    last_activity_detected_time = 0
    activity_detected_intervals = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    output_file = f"{video_name}_{timestamp}_output.txt"
    with open(output_file, "w") as f:
        def worker(frame_queue, f):
            nonlocal processed_frames, active_times, previous_boxes, last_activity_detected_time, activity_detected_intervals
            while True:
                try:
                    frame_num, frame = frame_queue.get(timeout=1)
                    current_boxes = process_frame(frame, model, transform, threshold)
                    if current_boxes:
                        with lock:
                            if last_activity_detected_time == 0:
                                last_activity_detected_time = frame_num
                            if previous_boxes and calculate_motion(previous_boxes, current_boxes):
                                active_times += interval
                                timestamp = format_time(frame_num / fps)
                                f.write(f"Activity detected at {timestamp}, duration: {interval} seconds\n")
                            previous_boxes = current_boxes
                    else:
                        if last_activity_detected_time != 0:
                            with lock:
                                activity_detected_intervals.append((last_activity_detected_time, frame_num))
                                last_activity_detected_time = 0
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
        # 记录最终结果
        video_duration_formatted = format_time(video_duration)
        active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0
        f.write(f"\nTotal active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration).\n")
        # 输出每个检测到活动的区间
        for start, end in activity_detected_intervals:
            start_time_str = format_time(start / fps)
            end_time_str = format_time(end / fps)
            f.write(f"Activity detected from {start_time_str} to {end_time_str}\n")
        print(f"\nProcessing frames: {processed_frames}/{total_frames} (100.00%)")
        print(f"Video analysis completed. Output written to {output_file}")
        return active_times, fps

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

# GUI setup
root = tk.Tk()
root.title("Video Analysis")
# 调整窗口默认大小
root.geometry("800x700")  # 可以根据需要调整宽度
root.resizable(True, True)  # 允许用户调整窗口大小

# Directory selection
directory_path = tk.StringVar()
frame_directory = ttk.Frame(root)
frame_directory.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
browse_button = ttk.Button(frame_directory, text="选择目录", command=lambda: browse_directory(directory_path, update_video_list))
browse_button.pack(side="left")
directory_label = ttk.Label(frame_directory, textvariable=directory_path, width=50)
directory_label.pack(side="left", padx=(10, 0))  # Add padding

# Function to select or deselect all videos
def select_deselect_all(select):
    for var, _, _, _ in video_checkboxes:
        var.set(select)

def analyze_single_video(var, file, result_entry, result_text):
    if var.get() and os.path.isfile(file):
        # 如果是 MTS 文件，则转换为 MP4
        if file.lower().endswith(".mts"):
            print("MTS file detected. Converting to MP4...")
            converted_path = convert_to_mp4(file)
            if not converted_path:
                result_text.set("转换失败。")
                return
            file = converted_path

        # 执行物体检测和运动分析
        active_times, fps = detect_objects_in_video(file, model, transform)
        if active_times >= 0:
            formatted_time = format_time(active_times)
            cap = cv2.VideoCapture(file)
            total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            cap.release()

            activity_ratio = (active_times / total_duration) * 100
            result_text.set(f"活动时间: {formatted_time}, ({active_times}秒), 活动占比: {activity_ratio:.2f}%")
        else:
            result_text.set("处理出错。")

def analyze_all_videos():
    total_active_times = 0
    total_videos_duration = 0
    total_videos = len(video_checkboxes)
    selected_videos_count = 0
    for var, file, result_entry, result_text in video_checkboxes:
        if var.get():  # 使用 var.get() 来检查选中状态
            selected_videos_count += 1
            full_file_path = os.path.join(directory_path.get(), file)  # 获取完整的文件路径
            active_times, _ = detect_objects_in_video(full_file_path, model, transform)
            if active_times >= 0:
                formatted_time = format_time(active_times)
                cap = cv2.VideoCapture(full_file_path)
                total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                total_videos_duration += total_duration
                result_text.set(f"活动时间: {formatted_time}, ({active_times}秒)")
                total_active_times += active_times
            else:
                result_text.set("处理出错。")

    if selected_videos_count > 0:
        average_active_time = total_active_times / selected_videos_count
        overall_activity_ratio = (total_active_times / total_videos_duration) * 100
        percentage = round(selected_videos_count / total_videos * 100, 2)
        mass_result.set(f"所有选定视频分析完成。\n"
                        f"总活动时间: {format_time(total_active_times)}, ({total_active_times}秒)\n"
                        f"平均每视频活动时间: {format_time(average_active_time)}, ({average_active_time}秒)\n"
                        f"已分析视频占比: {percentage}%\n"
                        f"活动时间占比: {overall_activity_ratio:.2f}%")
    elif total_videos == 0:
        mass_result.set("没有找到任何视频。")
    else:
        mass_result.set("没有选择任何视频。")

# Video list
frame_videos = ttk.Frame(root)
frame_videos.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
# 调整列权重，让中间的内容区域更宽
root.columnconfigure(0, weight=1)  # 给第0列分配更多权重
root.rowconfigure(1, weight=1)  # 给第1行分配更多权重，这是内容主要显示的地方

select_all_button = ttk.Button(frame_videos, text="全选", command=lambda: select_deselect_all(True))
select_all_button.pack(side="top", anchor="nw", pady=(0, 5))
deselect_all_button = ttk.Button(frame_videos, text="取消全选", command=lambda: select_deselect_all(False))
deselect_all_button.pack(side="top", anchor="nw")

video_frame = tk.Canvas(frame_videos)  # Use a Canvas for scrolling
video_frame.pack(side="left", fill="both", expand=True)
scrollbar = ttk.Scrollbar(frame_videos, orient="vertical", command=video_frame.yview)
scrollbar.pack(side="right", fill="y")
video_frame.configure(yscrollcommand=scrollbar.set)

inner_video_frame = ttk.Frame(video_frame)
video_frame.create_window((0, 0), window=inner_video_frame, anchor="nw")
inner_video_frame.bind("<Configure>", lambda e: video_frame.config(scrollregion=video_frame.bbox("all")))

# Mass analysis
frame_analysis = ttk.Frame(root)
frame_analysis.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
mass_result = tk.StringVar()
analyze_button = ttk.Button(frame_analysis, text="分析所有选定视频", command=analyze_all_videos)
analyze_button.pack(side="left", padx=(0, 10))

# Create a Text widget to display results with scrollbars
result_text = tk.Text(frame_analysis, wrap="word", height=4, width=60, relief="sunken")
result_text.pack(side="left", fill="both", expand=True)
vScrollbar = tk.Scrollbar(frame_analysis, orient="vertical", command=result_text.yview)
hScrollbar = tk.Scrollbar(frame_analysis, orient="horizontal", command=result_text.xview)
result_text.configure(xscrollcommand=hScrollbar.set, yscrollcommand=vScrollbar.set)
vScrollbar.pack(side="right", fill="y")
hScrollbar.pack(side="bottom", fill="x")

mass_result_label = ttk.Label(frame_analysis, textvariable=mass_result)
mass_result_label.pack(side="left")

# Function to browse for a directory and update the video list
def browse_directory(directory_var, update_func):
    directory = filedialog.askdirectory()
    if directory:
        directory_var.set(directory)
        update_func(directory)

# 在update_video_list函数中，增加Entry组件的宽度
def update_video_list(directory):
    for widget in inner_video_frame.winfo_children():
        widget.destroy()
    video_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.mp4', '.mts', '.avi'))]
    global video_checkboxes
    video_checkboxes = []
    for i, video_file in enumerate(video_files, start=1):
        var = tk.BooleanVar(value=False)
        checkbutton = ttk.Checkbutton(inner_video_frame, variable=var, text=video_file)
        checkbutton.grid(row=i, column=0, sticky="w")
        result_text = tk.StringVar()
        # 这里增加Entry组件的宽度
        result_entry = ttk.Entry(inner_video_frame, textvariable=result_text, state='readonly', width=80)  # 假设设置为50字符宽
        result_entry.grid(row=i, column=2, sticky="ew", padx=(10, 0))
        analyze_button = ttk.Button(inner_video_frame, text="分析", command=lambda v=var, f=os.path.join(directory, video_file), rt=result_text: analyze_single_video(v, f, None, rt))
        analyze_button.grid(row=i, column=1, padx=(10, 0))
        video_checkboxes.append((var, video_file, result_entry, result_text))

# 移除'分析所有选定视频'按钮后的白色文本框
# 首先删除旧的Text和Scrollbar组件
for widget in frame_analysis.winfo_children():
    widget.destroy()

# 重新创建mass_result_label
mass_result_label = ttk.Label(frame_analysis, textvariable=mass_result)
mass_result_label.pack(side="left")

# 重新创建analyze_button
analyze_button = ttk.Button(frame_analysis, text="分析所有选定视频", command=analyze_all_videos)
analyze_button.pack(side="left", padx=(0, 10))

# Start the main loop
root.mainloop()

# Helper functions
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"

# The rest of the helper functions (estimate_remaining_time, frame_generator, process_frame, calculate_motion, convert_to_mp4) are already defined above.
