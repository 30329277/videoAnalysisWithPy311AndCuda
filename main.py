import torch
from torchvision import models, transforms
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock

# 加载预训练的Faster R-CNN模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

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
    # 将帧转换为模型所需的格式
    frame_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(frame_tensor)
    
    # 遍历所有预测结果
    for pred in range(len(predictions[0]['labels'])):
        label_id = predictions[0]['labels'][pred].item()
        # 如果预测得分大于阈值，则返回对应的类别名称
        # if predictions[0]['scores'][pred] > threshold:
        if predictions[0]['labels'][pred].item() == 3 and predictions[0]['scores'][pred] > threshold:            
            return COCO_INSTANCE_CATEGORY_NAMES[label_id]
    # 如果没有找到符合条件的对象，则返回None
    return None

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
    detected_frames = {}  # 用于存储每个对象在哪些帧中被检测到
    lock = Lock()
    processed_frames = 0
    progress_lock = Lock()
    start_time = time.time()

    def worker(frame_queue):
        nonlocal processed_frames
        while True:
            try:
                frame_num, frame = frame_queue.get(timeout=1)
                object_name = process_frame(frame, model, transform, threshold)
                if object_name:
                    with lock:
                        if object_name not in detected_frames:
                            detected_frames[object_name] = []
                        detected_frames[object_name].append(frame_num)
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

    for object_name, frames in detected_frames.items():
        percentage = (len(frames) / total_frames) * 100
        print(f"{object_name} detected in {len(frames)} frames, which is {percentage:.2f}% of the total video duration.")

    return detected_frames, fps

def write_results_to_file(detected_frames, fps, output_file='output.txt'):
    with open(output_file, 'w') as f:
        for object_name, frames in detected_frames.items():
            for frame_num in frames:
                start_time = frame_num / fps
                end_time = (frame_num + 1) / fps
                duration = end_time - start_time
                f.write(f"{object_name} detected at {start_time:.2f}s, duration: {duration:.2f}s\n")

if __name__ == "__main__":
    video_path = "data/Media1.mp4"
    detected_frames, video_fps = detect_objects_in_video(video_path, model, transform, num_threads=4)
    write_results_to_file(detected_frames, video_fps)