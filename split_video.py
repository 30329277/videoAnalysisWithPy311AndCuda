import os
import subprocess

# 输入视频路径和输出目录
input_video_path = r"C:\local\IE4.0\Python\videoAnalysisWithPy311AndCuda\data\03331.MTS"  # 使用原始字符串，替换为你的 MTS 视频路径
output_dir = "split_video"
os.makedirs(output_dir, exist_ok=True)

# 定义每个视频片段的时长（单位：秒）
segment_duration = 3 * 60  # 3分钟

# ffmpeg 可执行文件路径
ffmpeg_path = r"ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

def split_video(input_path, output_directory, segment_length):
    # 获取视频文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 构建输出模板路径
    output_template = os.path.join(output_directory, f"{base_name}_%03d.mp4")
    
    # 构建 ffmpeg 命令
    command = [
        ffmpeg_path,
        "-i", input_path,  # 确保路径格式正确
        "-c", "copy",
        "-map", "0:0",  # 只复制第一个视频流
        "-map", "0:1",  # 只复制第一个音频流
        "-segment_time", str(segment_length),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_template
    ]
    
    # 执行命令
    try:
        subprocess.run(command, check=True)
        print(f"视频已成功分割并保存在 {output_directory} 文件夹中")
    except subprocess.CalledProcessError as e:
        print(f"视频分割失败: {e}")

# 调用函数进行视频分割
split_video(input_video_path, output_dir, segment_duration)
