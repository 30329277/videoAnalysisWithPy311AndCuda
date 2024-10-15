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

def detect_objects_in_video(video_path, model, transform, interval=0.1, num_threads=4, threshold=0.5):
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
    
    output_file = "output.txt"
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
    with open(output_file, "a") as f:
        f.write(f"\nTotal active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration).\n")
    
    return active_times, active_percentage, fps

# GUI 部分
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Activity Detection")
        self.geometry("400x200")

        self.label = ttk.Label(self, text="Select a video file to analyze.")
        self.label.pack(pady=10)

        self.button = ttk.Button(self, text="Open Video", command=self.open_video)
        self.button.pack(pady=10)

        self.result_label = ttk.Label(self, text="")
        self.result_label.pack(pady=10)

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if file_path:
            self.label.config(text="Processing...")
            self.update_idletasks()
            active_times, active_percentage, fps = detect_objects_in_video(file_path, model, transform)
            if active_times >= 0:
                result_text = f"Total active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration)."
                self.result_label.config(text=result_text)
            else:
                self.result_label.config(text="An error occurred during processing.")

if __name__ == "__main__":
    app = App()
    app.mainloop()