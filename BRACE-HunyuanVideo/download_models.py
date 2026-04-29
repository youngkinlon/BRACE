import os
import cv2
import numpy as np
import pandas as pd
import torch
import lpips
import shutil  # 👈 新增：用于自动拷贝视频文件
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ================= ⚙️ 配置区域 =================
# 1. 你的三个视频文件夹路径
ORIGIN_DIR = "Origin_test_N1"  # 无损基准 (Ground Truth)
BARY_DIR = "Bary_mini_test_N6"  # 你的 Barycentric 方法
TAYLOR_DIR = "Taylor_mini_test_N6"  # 对标的 Taylor 方法

# 2. 我们用来排名的核心终极指标 (推荐用 'LPIPS')
RANK_METRIC = 'LPIPS'

# 3. 提取出的 Top10 视频存放位置 👈 新增配置
OUT_SAMPLES_DIR = "Top10_Visual_Samples"
# ===============================================

# 自动检测并使用 GPU 极速计算 LPIPS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 正在使用计算设备: {device}")

# 加载 LPIPS 模型
print("📦 正在加载 LPIPS 感知模型...")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)


def get_video_metrics(test_video, gt_video):
    """逐帧对比两个视频，计算平均 PSNR, SSIM 和 LPIPS"""
    cap_test = cv2.VideoCapture(test_video)
    cap_gt = cv2.VideoCapture(gt_video)

    if not cap_test.isOpened() or not cap_gt.isOpened():
        return None, None, None

    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        while True:
            ret_test, frame_test = cap_test.read()
            ret_gt, frame_gt = cap_gt.read()

            if not ret_test or not ret_gt:
                break

            if frame_test.shape != frame_gt.shape:
                frame_test = cv2.resize(frame_test, (frame_gt.shape[1], frame_gt.shape[0]))

            # 1. 计算 SSIM
            gray_test = cv2.cvtColor(frame_test, cv2.COLOR_BGR2GRAY)
            gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
            psnr_val = compute_psnr(frame_gt, frame_test)
            ssim_val = compute_ssim(gray_gt, gray_test, data_range=255, win_size=11)

            # 2. 计算 LPIPS
            rgb_test = cv2.cvtColor(frame_test, cv2.COLOR_BGR2RGB)
            rgb_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)

            t_test = torch.from_numpy(rgb_test).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
            t_gt = torch.from_numpy(rgb_gt).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

            lpips_val = loss_fn_alex(t_gt, t_test).item()

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)

    cap_test.release()
    cap_gt.release()

    if not psnr_list:
        return 0, 0, 0

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


def main():
    print("🔍 正在逐帧解析视频像素，硬核比对 PSNR, SSIM 与 LPIPS...")

    origin_videos = [f for f in os.listdir(ORIGIN_DIR) if f.endswith('.mp4')]
    results = []

    total = len(origin_videos)
    for idx, vid_name in enumerate(origin_videos):
        orig_path = os.path.join(ORIGIN_DIR, vid_name)
        bary_path = os.path.join(BARY_DIR, vid_name)
        taylor_path = os.path.join(TAYLOR_DIR, vid_name)

        if not os.path.exists(bary_path) or not os.path.exists(taylor_path):
            continue

        print(f"🎬 进度 [{idx + 1}/{total}]: 计算 {vid_name[:30]}...")

        bary_psnr, bary_ssim, bary_lpips = get_video_metrics(bary_path, orig_path)
        taylor_psnr, taylor_ssim, taylor_lpips = get_video_metrics(taylor_path, orig_path)

        if bary_psnr is None or taylor_psnr is None:
            continue

        psnr_delta = bary_psnr - taylor_psnr
        ssim_delta = bary_ssim - taylor_ssim
        lpips_delta = taylor_lpips - bary_lpips

        results.append({
            "Video": vid_name,
            "Bary_LPIPS": bary_lpips,
            "Taylor_LPIPS": taylor_lpips,
            "LPIPS_Delta": lpips_delta,
            "Bary_PSNR": bary_psnr,
            "Taylor_PSNR": taylor_psnr,
            "PSNR_Delta": psnr_delta,
            "Bary_SSIM": bary_ssim,
            "Taylor_SSIM": taylor_ssim,
            "SSIM_Delta": ssim_delta
        })

    if not results:
        print("❌ 没有找到同名的测试视频！")
        return

    sort_key = f"{RANK_METRIC}_Delta"
    results.sort(key=lambda x: x[sort_key], reverse=True)

    top_10 = results[:10]

    print("\n" + "🌟" * 35)
    print(f"🔥 Barycentric 在视觉感知 ({RANK_METRIC}) 上碾压 Taylor 的 Top 10 瞬间 🔥")
    print("🌟" * 35 + "\n")

    df = pd.DataFrame(top_10)
    print_df = df[["Video", "LPIPS_Delta", "PSNR_Delta", "SSIM_Delta", "Bary_LPIPS", "Taylor_LPIPS"]]
    print(print_df.to_string(index=True, formatters={
        'LPIPS_Delta': '+{:.4f}'.format,
        'PSNR_Delta': '+{:.2f} dB'.format,
        'SSIM_Delta': '+{:.4f}'.format,
        'Bary_LPIPS': '{:.4f}'.format,
        'Taylor_LPIPS': '{:.4f}'.format
    }))

    csv_name = "Full_Metrics_LPIPS_PSNR_SSIM.csv"
    pd.DataFrame(results).to_csv(csv_name, index=False)
    print(f"\n💾 完整的像素级与感知对决数据已保存至: {csv_name}")

    # ========================================================
    # 🌟 自动提取并打包 Top 10 视频的逻辑 🌟
    # ========================================================
    print(f"\n📂 正在自动提取 Top 10 视频至 [{OUT_SAMPLES_DIR}] 文件夹...")
    os.makedirs(OUT_SAMPLES_DIR, exist_ok=True)

    for rank, row in enumerate(top_10):
        vid_name = row["Video"]
        safe_name = vid_name.replace(".mp4", "")
        # 为每个名次建一个独立的子文件夹
        subfolder_path = os.path.join(OUT_SAMPLES_DIR, f"Top{rank + 1}_{safe_name[:40]}")
        os.makedirs(subfolder_path, exist_ok=True)

        # 原始文件路径
        orig_src = os.path.join(ORIGIN_DIR, vid_name)
        bary_src = os.path.join(BARY_DIR, vid_name)
        taylor_src = os.path.join(TAYLOR_DIR, vid_name)

        # 目标文件路径 (带上分数标识，1_ 2_ 3_ 是为了在系统里按顺序排好)
        orig_dst = os.path.join(subfolder_path, f"1_Origin_GT.mp4")
        bary_dst = os.path.join(subfolder_path, f"2_Bary_[LPIPS_{row['Bary_LPIPS']:.3f}].mp4")
        taylor_dst = os.path.join(subfolder_path, f"3_Taylor_[LPIPS_{row['Taylor_LPIPS']:.3f}].mp4")

        # 拷贝文件
        if os.path.exists(orig_src): shutil.copy(orig_src, orig_dst)
        if os.path.exists(bary_src): shutil.copy(bary_src, bary_dst)
        if os.path.exists(taylor_src): shutil.copy(taylor_src, taylor_dst)

    print("✅ 提取完成！")
    print("👉 强烈建议把整个包下载到本地，用视频播放器左右开弓对比，挑出最让你震撼的几组！")


if __name__ == "__main__":
    main()