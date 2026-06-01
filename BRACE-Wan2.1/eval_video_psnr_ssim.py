"""
视频对比脚本：逐帧计算 PSNR 和 SSIM
用法：
    python eval_video_psnr_ssim.py \
        --gt_dir ./origin_videos \
        --target_dir ./chronomagic_outputs_taylor \
        --output results.json
"""
import os
import json
import argparse
import cv2
import numpy as np


# ==================== PSNR / SSIM 实现 ====================

def calculate_psnr(img1, img2, test_y_channel=False):
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}.'
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}.'
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


# ==================== 视频处理 ====================

def extract_frames(video_path, max_frames=None):
    """从视频提取所有帧"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def compare_videos(gt_path, target_path, sample_every=1, max_frames=None):
    """
    逐帧对比两个视频，返回每帧的 PSNR/SSIM。
    sample_every: 每隔N帧采样一次（1=全部帧，2=每2帧取1帧，加快计算）
    """
    gt_frames = extract_frames(gt_path, max_frames)
    target_frames = extract_frames(target_path, max_frames)

    n_frames = min(len(gt_frames), len(target_frames))
    if n_frames == 0:
        return None

    psnrs, ssims = [], []
    for i in range(0, n_frames, sample_every):
        gt_frame = gt_frames[i]
        target_frame = target_frames[i]

        # 如果尺寸不一致，resize target 到 gt 尺寸
        if gt_frame.shape != target_frame.shape:
            target_frame = cv2.resize(target_frame, (gt_frame.shape[1], gt_frame.shape[0]))

        psnr_val = calculate_psnr(gt_frame, target_frame, test_y_channel=True)
        ssim_val = calculate_ssim(gt_frame, target_frame, test_y_channel=True)
        psnrs.append(psnr_val)
        ssims.append(ssim_val)

    return {
        "frame_count": n_frames,
        "sampled_frames": len(psnrs),
        "psnr_avg": float(np.mean(psnrs)),
        "ssim_avg": float(np.mean(ssims)),
        "psnr_per_frame": [float(v) for v in psnrs],
        "ssim_per_frame": [float(v) for v in ssims],
    }


def main():
    parser = argparse.ArgumentParser(description="视频对比评测：PSNR & SSIM")
    parser.add_argument("--gt_dir", type=str, required=True, help="参考视频目录（原版模型生成）")
    parser.add_argument("--target_dir", type=str, required=True, help="待评测视频目录")
    parser.add_argument("--sample_every", type=int, default=5, help="每隔N帧采样（默认1=全部帧）")
    parser.add_argument("--max_frames", type=int, default=None, help="每个视频最多比较多少帧")
    parser.add_argument("--output", type=str, default="video_psnr_ssim.json", help="结果输出文件")
    args = parser.parse_args()

    gt_videos = sorted(os.listdir(args.gt_dir))
    target_videos = sorted(os.listdir(args.target_dir))

    # 匹配策略：从 target 里找 gt 中同名的视频
    gt_set = set(gt_videos)
    target_set = set(target_videos)
    common = sorted(gt_set & target_set)

    if not common:
        print(f"警告: 两个目录没有同名视频文件，尝试按顺序匹配...")
        common = sorted(target_videos)
        gt_lookup = {i: v for i, v in enumerate(sorted(gt_videos))}

    print(f"GT 目录: {args.gt_dir} ({len(gt_videos)} 个文件)")
    print(f"Target 目录: {args.target_dir} ({len(target_videos)} 个文件)")
    print(f"匹配到 {len(common)} 对视频")
    print("-" * 60)

    results = {}
    total_psnr, total_ssim, count = 0, 0, 0

    for name in common:
        gt_path = os.path.join(args.gt_dir, name)
        target_path = os.path.join(args.target_dir, name)
        print(f"对比: {name} ... ", end="", flush=True)

        res = compare_videos(gt_path, target_path, args.sample_every, args.max_frames)
        if res is None:
            print("失败 (无有效帧)")
            continue

        results[name] = res
        total_psnr += res["psnr_avg"]
        total_ssim += res["ssim_avg"]
        count += 1
        print(f"PSNR={res['psnr_avg']:.2f}, SSIM={res['ssim_avg']:.4f}")

    print("-" * 60)
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    print(f"整体平均: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}  ({count} 个视频)")

    # 保存结果
    output_data = {
        "gt_dir": args.gt_dir,
        "target_dir": args.target_dir,
        "video_count": count,
        "average_psnr": avg_psnr,
        "average_ssim": avg_ssim,
        "per_video": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
