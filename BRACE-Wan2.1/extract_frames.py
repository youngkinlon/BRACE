"""
视频抽帧脚本
用法：
    python extract_frames.py --input video.mp4 --output_dir frames/ --sample_every 1
"""
import os
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="从视频提取帧")
    parser.add_argument("--input", type=str, default="chronomagic_outputs_ori/fireworks_18.mp4", help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="frames_origin", help="输出图片目录")
    parser.add_argument("--sample_every", type=int, default=1, help="每隔N帧取1帧（1=全部帧）")
    parser.add_argument("--max_frames", type=int, default=None, help="最多提取多少帧")
    parser.add_argument("--prefix", type=str, default="frame", help="输出文件名前缀")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")

    saved_count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.sample_every == 0:
            out_path = os.path.join(args.output_dir, f"{args.prefix}_{frame_idx:06d}.png")
            cv2.imwrite(out_path, frame)
            saved_count += 1

            if args.max_frames and saved_count >= args.max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"完成: 共保存 {saved_count} 帧到 {args.output_dir}/")


if __name__ == "__main__":
    main()