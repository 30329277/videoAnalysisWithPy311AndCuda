import torch
from torchvision import models, transforms
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock
from collections import defaultdict

# 加载预训练的Faster R-CNN模型，并将模型移至GPU
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to('cuda')

# COCO数据集中类别的名称列表
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

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def process_frame(frame, model, transform, threshold=0.8):
    frame_tensor = transform(frame).unsqueeze(0).to('cuda')
    with torch.no_grad():
        predictions = model(frame_tensor)

    objects = []
    for idx in range(len(predictions[0]['labels'])):
        score = predictions[0]['scores'][idx].item()
        if score > threshold:
            label_id = predictions[0]['labels'][idx].item()
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
            bbox = predictions[0]['boxes'][idx].tolist()  # 获取边界框位置
            objects.append((label_id, label_name, bbox, score))
    return objects

def frame_generator(video_path, target_width=640, target_height=480):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
    cap.release()

def estimate_remaining_time(start_time, processed_frames, total_frames, fps):
    elapsed_time = time.time() - start_time
    avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0
    remaining_frames = total_frames - processed_frames
    remaining_time = remaining_frames * avg_time_per_frame
    return remaining_time

def detect_objects_in_video(video_path, model, transform, num_threads=4, threshold=0.8):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_frames = []  # 用于存储检测到的对象及其信息
    lock = Lock()
    processed_frames = 0
    progress_lock = Lock()
    start_time = time.time()

    def worker(frame_queue):
        nonlocal processed_frames
        while True:
            try:
                frame_num, frame = frame_queue.get(timeout=1)
                objects = process_frame(frame, model, transform, threshold)
                if objects:
                    with lock:
                        for obj in objects:
                            detected_frames.append((frame_num, obj))
                with progress_lock:
                    processed_frames += 1
                frame_queue.task_done()
            except Queue.Empty:
                break

    frame_queue = Queue(maxsize=num_threads * 10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(worker, frame_queue)

        for frame_num, frame in frame_generator(video_path):
            frame_queue.put((frame_num, frame))

            with progress_lock:
                current_progress = (processed_frames / total_frames) * 100
                remaining_time = estimate_remaining_time(start_time, processed_frames, total_frames, fps)
                print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%), Estimated time remaining: {remaining_time:.2f}s", end='\r')

    frame_queue.join()

    with progress_lock:
        current_progress = (processed_frames / total_frames) * 100
    print(f"\nProcessing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%)")
    end_time = time.time()
    print(f"Video analysis completed. Time taken: {end_time - start_time:.2f} seconds")

    return detected_frames, fps

def write_results_to_file(detected_frames, fps, output_file='output.txt'):
    object_durations = defaultdict(float)
    total_video_duration = max(frame_num for frame_num, _ in detected_frames) / fps

    with open(output_file, 'w') as f:
        for frame_num, obj in detected_frames:
            label_id, label_name, bbox, score = obj
            start_time = frame_num / fps
            end_time = (frame_num + 1) / fps
            duration = end_time - start_time
            bbox_str = ', '.join(f"{coord:.2f}" for coord in bbox)
            f.write(f"ID: {label_id}, Name: {label_name}, BBox: [{bbox_str}], Start: {start_time:.2f}s, Duration: {duration:.2f}s, Score: {score:.2f}\n")
            
            # 累加每种对象的总时长
            object_durations[label_name] += duration

        # 写入汇总信息
        f.write("\nSummary of object durations:\n")
        for label_name, total_duration in object_durations.items():
            percentage = (total_duration / total_video_duration) * 100
            f.write(f"{label_name}: Total Duration = {total_duration:.2f}s, Percentage = {percentage:.2f}%\n")

if __name__ == "__main__":
    video_path = "data/Media1.mp4"
    detected_frames, video_fps = detect_objects_in_video(video_path, model, transform, num_threads=4)
    write_results_to_file(detected_frames, video_fps)
