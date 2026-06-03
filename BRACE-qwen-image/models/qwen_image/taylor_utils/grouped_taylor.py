"""
K-means based dimension-wise TaylorSeer implementation.
将D维特征按照相似性聚类，不同聚类组使用不同的TaylorSeer阶数。
"""

import torch
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from datetime import datetime

# 可选导入matplotlib和seaborn
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def simple_kmeans(X, n_clusters, max_iters=100, random_state=42):
    """
    简单的K-means实现，避免sklearn依赖
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # 随机初始化聚类中心
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iters):
        # 计算每个点到各个聚类中心的距离
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        
        # 分配到最近的聚类中心
        labels = np.argmin(distances, axis=0)
        
        # 更新聚类中心
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(n_clusters)])
        
        # 检查收敛
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return labels


class DimensionGroupedTaylorSeer:
    """
    基于K-means聚类的维度分组TaylorSeer预测器
    """
    
    def __init__(self, 
                 feature_dim: int,
                 n_clusters: int = 4,
                 orders: List[int] = [0, 1, 2, 3],
                 history_window: int = 4):  # 改为4步以支持三阶差分
        """
        Args:
            feature_dim: 特征维度D
            n_clusters: 聚类数量
            orders: 每个聚类组对应的TaylorSeer阶数（仅作为初始值，实际会动态计算）
            history_window: 用于聚类的历史窗口大小（至少4步）
        """
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.orders = orders[:n_clusters] if orders else [0] * n_clusters  # 初始阶数，会被动态更新
        self.history_window = max(4, history_window)  # 至少4步来计算三阶差分
        
        # 维度分组
        self.dimension_groups = None  # 每个维度所属的组
        self.is_fitted = False
        
        # 历史特征用于聚类分析
        self.feature_history = []

        # 记录聚类次数
        self.cluster_count = 0
        
    def _collect_dimension_statistics(self, features: List[torch.Tensor]) -> Optional[np.ndarray]:
        """
        收集每个维度的数值分析特征用于聚类
        基于时间步的差分计算和频域分析
        
        Args:
            features: 历史特征列表 [N, D], 按时间步排序
            
        Returns:
            dimension_stats: [D, n_stats] 每个维度的数值分析特征
        """
        if len(features) < 2:
            return None

        # 转换为numpy并计算统计量（先转为 float32，避免 bfloat16 无法转换的问题）
        features_np = [f.detach().cpu().to(dtype=torch.float32).numpy() for f in features]
        
        # 检查所有特征是否具有相同形状
        first_shape = features_np[0].shape
        if not all(f.shape == first_shape for f in features_np):
            # print(f"[GroupedTaylor] Warning: Feature shapes inconsistent: {[f.shape for f in features_np]}")
            # 找到最小的公共形状，保持3维结构 [batch, seq_len, feature_dim]
            if len(first_shape) == 3:
                min_batch = min(f.shape[0] for f in features_np)
                min_seq = min(f.shape[1] for f in features_np)
                min_feat = min(f.shape[2] for f in features_np)
                features_np = [f[:min_batch, :min_seq, :min_feat] for f in features_np]
                # print(f"[GroupedTaylor] Trimmed to common shape: ({min_batch}, {min_seq}, {min_feat})")
            elif len(first_shape) == 2:
                min_n = min(f.shape[0] for f in features_np)
                min_d = min(f.shape[1] for f in features_np)
                features_np = [f[:min_n, :min_d] for f in features_np]
                # print(f"[GroupedTaylor] Trimmed to common shape: ({min_n}, {min_d})")
        
        stacked_features = np.stack(features_np, axis=0)  # [T, batch, seq_len, feature_dim] or [T, N, D]
        
        # 处理不同的维度情况
        if len(stacked_features.shape) == 4:  # [T, batch, seq_len, feature_dim]
            T, batch, seq_len, D = stacked_features.shape
            # 重新整形为 [T, N, D] 其中 N = batch * seq_len
            stacked_features = stacked_features.reshape(T, batch * seq_len, D)
            T, N, D = stacked_features.shape
        elif len(stacked_features.shape) == 3:  # [T, N, D]
            T, N, D = stacked_features.shape
        else:
            #print(f"[GroupedTaylor] Error: Unexpected feature shape {stacked_features.shape}")
            return None
        
        stats_list = []
        eps = 1e-8
        
        # 对每个维度计算数值分析特征
        for d in range(D):
            v = stacked_features[:, :, d]  # [T, N] - 第d维在各时间步的token向量
            
            # 初始化特征
            d1 = d2 = d3 = 0.0  # 一阶、二阶、三阶变化
            eta = kappa = rho = gamma = e = 0.0  # 曲率比、jerk比、方向一致性、相对变化率、能量
            lfr = sf = 0.0  # 低频占比、谱平坦度
            
            # 当前时间步（最新）
            vk = v[-1]  # [N]
            e = np.linalg.norm(vk)  # 能量
            
            if T >= 2:
                vk_1 = v[-2]  # [N]
                # 一阶变化（速度）
                d1 = np.linalg.norm(vk - vk_1)
                
                # 相对变化率
                gamma = d1 / (np.linalg.norm(vk_1) + eps)
                
                if T >= 3:
                    vk_2 = v[-3]  # [N]
                    # 二阶变化（加速度/曲率）
                    d2 = np.linalg.norm(vk - 2*vk_1 + vk_2)
                    
                    # 曲率比
                    eta = d2 / (d1 + eps)
                    
                    # 方向一致性（速度向量夹角）
                    vel1 = vk - vk_1
                    vel2 = vk_1 - vk_2
                    cos_angle = np.dot(vel1, vel2) / (np.linalg.norm(vel1) * np.linalg.norm(vel2) + eps)
                    rho = np.clip(cos_angle, -1.0, 1.0)
                    
                    if T >= 4:
                        vk_3 = v[-4]  # [N]
                        # 三阶变化（jerk）
                        d3 = np.linalg.norm(vk - 3*vk_1 + 3*vk_2 - vk_3)
                        
                        # jerk比
                        kappa = d3 / (d2 + eps)
            
            # 频域分析（对当前token向量做DCT）
            if N > 1:
                try:
                    # 简单的离散余弦变换近似
                    dct_coeffs = np.fft.fft(vk).real
                    dct_power = dct_coeffs ** 2
                    
                    # 低频占比（前20%频率）
                    q = max(1, N // 5)
                    lfr = np.sum(dct_power[:q]) / (np.sum(dct_power) + eps)
                    
                    # 谱平坦度（几何均值/算术均值）
                    nonzero_power = dct_power[dct_power > eps]
                    if len(nonzero_power) > 0:
                        geom_mean = np.exp(np.mean(np.log(nonzero_power)))
                        arith_mean = np.mean(nonzero_power)
                        sf = geom_mean / (arith_mean + eps)
                    else:
                        sf = 0.0
                        
                except:
                    # DCT计算失败时使用默认值
                    lfr = sf = 0.0
            
            # 组合所有数值分析特征
            # [d1, d2, d3, eta, kappa, rho, gamma, e, lfr, sf]
            dim_stats = [d1, d2, d3, eta, kappa, rho, gamma, e, lfr, sf]
            stats_list.append(dim_stats)
        
        return np.array(stats_list)  # [D, 10]
    
    def _assign_taylor_orders(self, dimension_groups: np.ndarray, dimension_stats: np.ndarray) -> List[int]:
        """
        基于数值分析特征为每个聚类组分配Taylor阶数
        根据 d2 和 d3 的大小智能选择算法：
        - d2 和 d3 较小时：使用 order=2 的 TaylorSeer 计算
        - d2 和 d3 较大时：使用 order=4 代指 FoCa 算法
        
        Args:
            dimension_groups: [D] 每个维度的聚类组ID
            dimension_stats: [D, 10] 每个维度的数值分析特征
                            [d1, d2, d3, eta, kappa, rho, gamma, e, lfr, sf]
        
        Returns:
            group_orders: List[int] 每个组的Taylor阶数 [0,1,2,3,4]
                         其中 4 表示使用 FoCa 算法
        """
        group_orders = []        
        for group_id in range(self.n_clusters):
            group_mask = (dimension_groups == group_id)
            if not np.any(group_mask):
                group_orders.append(0)
                continue
                
            # 提取该组的特征统计
            group_stats = dimension_stats[group_mask]  # [n_dims_in_group, 10]
            
            # 计算组级别的特征（取中位数或均值）
            d1, d2, d3, eta, kappa, rho, gamma, e, lfr, sf = np.median(group_stats, axis=0)
            
            # 定义 d2 和 d3 的阈值来判断是否使用 FoCa 算法
            d2_threshold = 70  # 可根据实际情况调整
            d3_threshold = 150  # 可根据实际情况调整
            
            # 首先判断是否需要使用 FoCa 算法
            if d2 < d2_threshold or d3 < d3_threshold:
                # d2 或 d3 较大，使用 FoCa 算法
                # print(f"Debug: Group {group_id} uses FoCa algorithm (order=4, d2={d2:.4f}, d3={d3:.4f})")
                order = 4  # 特殊标记，表示使用 FoCa 算法
            else:
                # d2 和 d3 较小，使用传统的阶数分配逻辑，默认使用 order=2
                order = 2
                
                # 3阶条件：放宽阈值，基于相对较好的稳定性
                # if (eta < 3.0 and kappa < 5.0 and rho > -0.1 and 
                #     gamma < 5.0 and e > np.percentile(dimension_stats[:, 7], 40)):
                #     print(f"Debug: Group {group_id} order set to 3 (d2={d2:.4f}, d3={d3:.4f})")
                #     order = 3
                    
                # # 2阶条件：中等曲率且相对稳定（默认选择）
                # elif (eta < 5.0 and kappa < 8.0 and rho > -0.3 and 
                #     gamma < 8.0):
                #     print(f"Debug: Group {group_id} order set to 2 (d2={d2:.4f}, d3={d3:.4f})")
                #     order = 2
                    
                # # 1阶条件：线性变化但可预测
                # elif (eta < 8.0 and rho > -0.6 and gamma < 12.0):
                #     print(f"Debug: Group {group_id} order set to 1 (d2={d2:.4f}, d3={d3:.4f})")
                #     order = 1
                    
                # # 0阶：高噪声、高变化、方向不稳定
                # else:
                #     print(f"Debug: Group {group_id} order set to 0 (d2={d2:.4f}, d3={d3:.4f})")
                #     order = 0
            
            group_orders.append(order)
        
        return group_orders

    def _save_dimension_stats(self, dimension_stats: np.ndarray, dimension_stats_scaled: np.ndarray,
                             dimension_stats_mean: np.ndarray, dimension_stats_std: np.ndarray,
                             save_path: str, image_idx: int):
        """
        保存维度统计数据用于后续分析
        
        Args:
            dimension_stats: [D, 10] 原始统计特征
            dimension_stats_scaled: [D, 10] 标准化后的统计特征
            dimension_stats_mean: [10] 各特征的均值
            dimension_stats_std: [10] 各特征的标准差
            save_path: 保存路径
            image_idx: 图像索引
        """
        try:
            # 创建保存目录
            stats_save_path = os.path.join(save_path, "dimension_stats")
            os.makedirs(stats_save_path, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'dimension_stats_img_{image_idx:04d}_{timestamp}.npz'
            full_path = os.path.join(stats_save_path, filename)
            
            # 特征名称，便于后续分析
            feature_names = ['d1', 'd2', 'd3', 'eta', 'kappa', 'rho', 'gamma', 'e', 'lfr', 'sf']
            
            # 保存数据
            save_data = {
                'dimension_stats_raw': dimension_stats,
                'dimension_stats_scaled': dimension_stats_scaled,
                'dimension_stats_mean': dimension_stats_mean,
                'dimension_stats_std': dimension_stats_std,
                'feature_names': np.array(feature_names),
                'image_idx': image_idx,
                'timestamp': timestamp,
                'n_clusters': self.n_clusters,
                'feature_dim': self.feature_dim
            }
            
            # 只有在聚类完成时才保存聚类结果
            if self.is_fitted and self.dimension_groups is not None:
                save_data['dimension_groups'] = self.dimension_groups
                save_data['group_orders'] = np.array(self.orders)
            
            np.savez_compressed(full_path, **save_data)
            
            print(f"[GroupedTaylor] Dimension stats saved: {full_path}")
            
            # 同时保存一个可读的文本摘要
            summary_filename = f'dimension_stats_summary_img_{image_idx:04d}_{timestamp}.txt'
            summary_path = os.path.join(stats_save_path, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"维度统计数据摘要 - 图像 {image_idx}\n")
                f.write(f"生成时间: {timestamp}\n")
                f.write(f"特征维度: {self.feature_dim}\n")
                f.write(f"聚类数量: {self.n_clusters}\n")
                f.write("\n特征说明:\n")
                feature_descriptions = [
                    "d1: 一阶变化（速度）",
                    "d2: 二阶变化（加速度/曲率）", 
                    "d3: 三阶变化（jerk）",
                    "eta: 曲率比 (d2/d1)",
                    "kappa: jerk比 (d3/d2)",
                    "rho: 方向一致性（速度向量夹角余弦值）",
                    "gamma: 相对变化率 (d1/||v_{k-1}||)",
                    "e: 能量 (||v_k||)",
                    "lfr: 低频占比（DCT前20%频率功率占比）",
                    "sf: 谱平坦度（几何均值/算术均值）"
                ]
                for desc in feature_descriptions:
                    f.write(f"  {desc}\n")
                
                f.write("\n原始统计特征范围:\n")
                for i, name in enumerate(feature_names):
                    min_val = np.min(dimension_stats[:, i])
                    max_val = np.max(dimension_stats[:, i])
                    mean_val = dimension_stats_mean[i]
                    std_val = dimension_stats_std[i]
                    f.write(f"  {name}: [{min_val:.6f}, {max_val:.6f}], mean={mean_val:.6f}, std={std_val:.6f}\n")
                
                if self.is_fitted:
                    f.write(f"\n聚类结果:\n")
                    f.write(f"  各组维度数量: {[np.sum(self.dimension_groups == i) for i in range(self.n_clusters)]}\n")
                    f.write(f"  各组Taylor阶数: {self.orders}\n")
            
            print(f"[GroupedTaylor] Dimension stats summary saved: {summary_path}")
            
        except Exception as e:
            print(f"[GroupedTaylor] Failed to save dimension stats: {e}")
            import traceback
            traceback.print_exc()

    def update_and_cluster(self, feature: torch.Tensor, save_visualization: bool = False, 
                          save_path: str = "/root/autodl-tmp/TaylorSeer/PCA-FLUX_copy/AnalyseResults", 
                          image_idx: int = 0) -> bool:
        """
        更新特征历史并尝试进行聚类
        
        Args:
            feature: 当前特征 [N, D]
            save_visualization: 是否保存可视化结果
            save_path: 可视化保存路径
            image_idx: 图像索引
            
        Returns:
            是否成功更新了聚类
        """
        # 添加到历史
        self.feature_history.append(feature.clone())

        # try:
        #     print(f"[GroupedTaylor] update_and_cluster called: history_len={len(self.feature_history)}, save_visualization={save_visualization}, image_idx={image_idx}")
        # except Exception:
        #     pass
        
        # 保持历史窗口大小
        if len(self.feature_history) > self.history_window:
            self.feature_history.pop(0)
        
        # 当有足够历史数据时进行聚类（至少需要4步来计算三阶差分）
        if len(self.feature_history) >= 4 and not self.is_fitted:
            dimension_stats = self._collect_dimension_statistics(self.feature_history)
            self.cluster_count += 1
            if dimension_stats is not None:
                # 标准化统计特征
                dimension_stats_mean = np.mean(dimension_stats, axis=0)
                dimension_stats_std = np.std(dimension_stats, axis=0)
                dimension_stats_std = np.where(dimension_stats_std == 0, 1, dimension_stats_std)  # 避免除零
                dimension_stats_scaled = (dimension_stats - dimension_stats_mean) / dimension_stats_std
                
                # 保存dimension_stats_scaled用于后续分析
                self._save_dimension_stats(dimension_stats, dimension_stats_scaled, 
                                         dimension_stats_mean, dimension_stats_std, 
                                         save_path, image_idx)

                # 只使用d3和eta作为聚类特征 (索引2和3) 
                clustering_features = dimension_stats_scaled[:, [2, 3]]

                # 执行K-means聚类
                self.dimension_groups = simple_kmeans(clustering_features, self.n_clusters)
                
                # 基于数值分析特征动态分配Taylor阶数
                self.orders = self._assign_taylor_orders(self.dimension_groups, dimension_stats)
                
                # if self.cluster_count >= 50:
                self.is_fitted = True
                # 注释为了保存每一步的npz，检查聚类是否有相似性
                
                # print(f"Dimension clustering completed: {self.n_clusters} groups, dimensions per group: {[np.sum(self.dimension_groups == i) for i in range(self.n_clusters)]}")
                # print(f"Taylor orders assigned to each group: {self.orders}")
                # try:
                #     print(f"[GroupedTaylor] Attempting to save visualization to {save_path} for image {image_idx}")
                # except Exception:
                #     pass
                
                # 输出每组的特征统计摘要
                for group_id in range(self.n_clusters):
                    group_mask = (self.dimension_groups == group_id)
                    if np.any(group_mask):
                        group_stats = dimension_stats[group_mask]
                        eta_med = np.median(group_stats[:, 3])  # 曲率比
                        kappa_med = np.median(group_stats[:, 4])  # jerk比
                        rho_med = np.median(group_stats[:, 5])  # 方向一致性
                        print(f"  Group{group_id} (Order{self.orders[group_id]}): eta={eta_med:.3f}, kappa={kappa_med:.3f}, rho={rho_med:.3f}")
                
                    # 保存可视化结果
                    if save_visualization:
                        self.visualize_clustering(dimension_stats, save_path, image_idx)
                
                return True
        
        return False
    
    def get_dimension_groups(self) -> Optional[np.ndarray]:
        """获取维度分组结果"""
        return self.dimension_groups if self.is_fitted else None
    
    def visualize_clustering(self, dimension_stats: np.ndarray, save_path: str, image_idx: int = 0):
        """
        可视化维度聚类结果并保存
        
        Args:
            dimension_stats: [D, 10] 每个维度的数值分析特征
            save_path: 保存路径
            image_idx: 图像索引
        """
        if not self.is_fitted or self.dimension_groups is None:
            return
            
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available, skipping visualization")
            return
            
        try:
            # 创建保存目录
            os.makedirs(save_path, exist_ok=True)
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Dimension Clustering Results - Image {image_idx}', fontsize=16, fontweight='bold')
            
            # 颜色映射
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            group_colors = [colors[i % len(colors)] for i in range(self.n_clusters)]
            
            # 1. 维度分组热图
            ax1 = axes[0, 0]
            dimension_matrix = np.zeros((1, self.feature_dim))
            for i, group_id in enumerate(self.dimension_groups):
                dimension_matrix[0, i] = group_id
            
            im1 = ax1.imshow(dimension_matrix, aspect='auto', cmap='tab10')
            plt.colorbar(im1, ax=ax1, label='Cluster Group ID')
            ax1.set_title('Dimension Group Heatmap')
            ax1.set_xlabel('Dimension Index')
            ax1.set_ylabel('Group Assignment')
            
            # 2. 各组的Taylor阶数分布
            ax2 = axes[0, 1]
            order_counts = [self.orders.count(i) for i in range(4)]
            bars = ax2.bar(range(4), order_counts, color=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon'])
            ax2.set_title('Taylor Order Distribution')
            ax2.set_xlabel('Taylor Order')
            ax2.set_ylabel('Number of Groups')
            ax2.set_xticks(range(4))
            # 添加数值标签
            for i, v in enumerate(order_counts):
                if v > 0:
                    ax2.text(i, v + 0.05, str(v), ha='center', va='bottom')
            
            # 3. 维度特征散点图 (曲率比 vs jerk比)
            ax3 = axes[0, 2]
            eta_values = dimension_stats[:, 3]  # 曲率比
            kappa_values = dimension_stats[:, 4]  # jerk比
            
            for group_id in range(self.n_clusters):
                group_mask = (self.dimension_groups == group_id)
                if np.any(group_mask):
                    ax3.scatter(eta_values[group_mask], kappa_values[group_mask], 
                              c=group_colors[group_id], label=f'G{group_id} (O{self.orders[group_id]})',
                              alpha=0.7, s=30)
            
            ax3.set_xlabel('Curvature Ratio (η)')
            ax3.set_ylabel('Jerk Ratio (κ)')
            ax3.set_title('Feature Space Distribution (η vs κ)')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # 4. 方向一致性分布
            ax4 = axes[1, 0]
            rho_values = dimension_stats[:, 5]  # 方向一致性
            
            for group_id in range(self.n_clusters):
                group_mask = (self.dimension_groups == group_id)
                if np.any(group_mask):
                    group_rho = rho_values[group_mask]
                    ax4.hist(group_rho, bins=15, alpha=0.6, color=group_colors[group_id], 
                           label=f'G{group_id} (O{self.orders[group_id]})')
            
            ax4.set_xlabel('Direction Consistency (ρ)')
            ax4.set_ylabel('Number of Dimensions')
            ax4.set_title('Direction Consistency Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. 能量分布
            ax5 = axes[1, 1]
            energy_values = dimension_stats[:, 7]  # 能量
            
            for group_id in range(self.n_clusters):
                group_mask = (self.dimension_groups == group_id)
                if np.any(group_mask):
                    group_energy = energy_values[group_mask]
                    ax5.hist(group_energy, bins=15, alpha=0.6, color=group_colors[group_id],
                           label=f'G{group_id} (O{self.orders[group_id]})')
            
            ax5.set_xlabel('Energy (e)')
            ax5.set_ylabel('Number of Dimensions')
            ax5.set_title('Energy Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. 组统计摘要表
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # 创建统计表格
            table_data = []
            for group_id in range(self.n_clusters):
                group_mask = (self.dimension_groups == group_id)
                if np.any(group_mask):
                    group_stats = dimension_stats[group_mask]
                    n_dims = np.sum(group_mask)
                    avg_eta = np.median(group_stats[:, 3])
                    avg_kappa = np.median(group_stats[:, 4])
                    avg_rho = np.median(group_stats[:, 5])
                    avg_energy = np.median(group_stats[:, 7])
                    
                    table_data.append([
                        f'G{group_id}',
                        f'O{self.orders[group_id]}',
                        f'{n_dims}',
                        f'{avg_eta:.3f}',
                        f'{avg_kappa:.3f}',
                        f'{avg_rho:.3f}',
                        f'{avg_energy:.3f}'
                    ])
            
            if table_data:
                table = ax6.table(cellText=table_data,
                                colLabels=['Group', 'Order', 'Dims', 'η', 'κ', 'ρ', 'Energy'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 2)
                ax6.set_title('Group Statistics Summary', pad=20)
            
            plt.tight_layout()
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'clustering_img_{image_idx:04d}_{timestamp}.png'
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # print(f"Clustering visualization saved: {full_path}")
            
        except Exception as e:
            print(f"Visualization save failed: {e}")

    def get_orders_for_groups(self) -> List[int]:
        """获取每个组对应的阶数"""
        return self.orders


def create_grouped_taylor_cache(cache_dic: Dict, current: Dict, feature_dim: int):
    """
    为分组TaylorSeer初始化缓存结构
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息
        feature_dim: 特征维度
    """
    if 'dimension_grouper' not in cache_dic:
        # 从配置中获取聚类参数
        # print("Debug: dimension_grouper not in cache dict")

        n_clusters = cache_dic.get('n_clusters', 4)
        orders = cache_dic.get('group_orders', [0, 1, 2, 3, 4])
        history_window = cache_dic.get('history_window', 5)
        
        cache_dic['dimension_grouper'] = DimensionGroupedTaylorSeer(
            feature_dim=feature_dim,
            n_clusters=n_clusters,
            orders=orders,
            history_window=history_window
        )
        # try:
        #     print(f"[GroupedTaylor] Created dimension_grouper: feature_dim={feature_dim}, n_clusters={n_clusters}, history_window={history_window}")
        # except Exception:
        #     pass
    # else:
        # print("Debug: dimension_grouper already exists in cache dict")

    # 初始化分组缓存
    if current['step'] == 0 and cache_dic['taylor_cache']:
        # print("Debug: current step is 0, initializing grouped_cache")
        layer_key = (current['model'], current['layer'], current['block'])
        if 'grouped_cache' not in cache_dic:
            # print("Debug: grouped_cache not in cache dict, initializing")
            cache_dic['grouped_cache'] = {}
        
        if layer_key not in cache_dic['grouped_cache']:
            # print(f"Debug: Initializing grouped_cache for layer_key {layer_key}")
            cache_dic['grouped_cache'][layer_key] = {}


def grouped_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    分组进行导数近似计算
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息  
        feature: 当前特征 [N, D]
    """
    # 初始化clustering_count
    if 'clustering_count' not in cache_dic:
        cache_dic['clustering_count'] = 0
    
    cache_dic['clustering_count'] += 1
    
    layer_key = (current['model'], current['layer'], current['block'])
    grouper = cache_dic['dimension_grouper']
    
    # 获取图像索引用于可视化
    image_idx = cache_dic.get('current_image_idx', 0)
    save_vis = cache_dic.get('save_clustering_visualization', False)
    

    if cache_dic.get('one_time_clustering', False):

        # 只有在聚类次数不超过8次时才进行聚类更新
        if cache_dic['clustering_count'] <= 8:
            # 更新历史并尝试聚类（带可视化选项）
            grouper.update_and_cluster(feature, save_visualization=save_vis, image_idx=image_idx)
        else:
            # 聚类已完成，只添加特征到历史但不进行新的聚类
            grouper.feature_history.append(feature.clone())
            if len(grouper.feature_history) > grouper.history_window:
                grouper.feature_history.pop(0)
    
    else:
        # 每次都更新并尝试聚类
        grouper.update_and_cluster(feature, save_visualization=save_vis, image_idx=image_idx)
    
    if not grouper.is_fitted:
        # 如果还没有聚类，使用原始方法
        # print("Debug: Grouper not fitted, using original TaylorSeer method")
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
        updated_taylor_factors = {0: feature}
        
        for i in range(cache_dic['max_order']):
            if (cache_dic['cache'][-1][current['model']][current['layer']][current['block']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
                updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['model']][current['layer']][current['block']][i]) / difference_distance
            else:
                break
        
        cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = updated_taylor_factors
        return
    
    # 使用分组方法
    dimension_groups = grouper.get_dimension_groups()
    group_orders = grouper.get_orders_for_groups()
    # print("Debug: group_orders =", group_orders)

    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    
    # 为每个组分别处理
    grouped_cache = cache_dic['grouped_cache'][layer_key]
    
    for group_id in range(grouper.n_clusters):
        group_mask = (dimension_groups == group_id)
        
        # 处理不同维度的特征张量
        if len(feature.shape) == 3:  # [batch, seq_len, feature_dim]
            group_feature = feature[:, :, group_mask]  # [batch, seq_len, D_group]
        elif len(feature.shape) == 2:  # [seq_len, feature_dim] or [N, D]
            group_feature = feature[:, group_mask]  # [seq_len, D_group] or [N, D_group]
        else:
            print(f"[GroupedTaylor] Error: Unsupported feature shape {feature.shape}")
            continue
            
        max_order = group_orders[group_id]
        
        if group_id not in grouped_cache:
            grouped_cache[group_id] = {}
        
        # 检查是否使用 FoCa 算法 (order=4)
        if max_order == 4:
            # 使用 FoCa 的缓存策略：缓存函数值序列
            from . import derivative_approximation_foca
            
            # 为 FoCa 算法准备 cache_dic 和 current 参数
            foca_current = current.copy()
            foca_cache_dic = {
                'cache': [{current['model']: {current['layer']: {current['block']: []}}}],
                'max_order': 2
            }
            
            # 如果已有缓存，转换为 FoCa 格式
            if len(grouped_cache[group_id]) > 0:
                # 从导数缓存转换为函数值序列缓存
                feats_list = []
                for i in range(min(4, len(grouped_cache[group_id]))):  # 最多保留4个历史值
                    if i in grouped_cache[group_id]:
                        feats_list.append(grouped_cache[group_id][i])
                foca_cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = feats_list
            
            try:
                derivative_approximation_foca(foca_cache_dic, foca_current, group_feature)
                # 将 FoCa 的缓存结果转换回我们的格式
                foca_feats = foca_cache_dic['cache'][-1][current['model']][current['layer']][current['block']]
                grouped_cache[group_id] = {i: foca_feats[i] for i in range(len(foca_feats))}
            except Exception as e:
                # print(f"[GroupedTaylor] FoCa caching failed for group {group_id}: {e}")
                # 回退到传统缓存方法
                updated_taylor_factors = {0: group_feature}
                for i in range(min(max_order, 3)):  # 限制最大阶数
                    if (grouped_cache[group_id].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
                        updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - grouped_cache[group_id][i]) / difference_distance
                    else:
                        break
                grouped_cache[group_id] = updated_taylor_factors
        else:
            # 使用传统的 Taylor 导数缓存方法
            updated_taylor_factors = {0: group_feature}
            
            for i in range(max_order):
                if (grouped_cache[group_id].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
                    updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - grouped_cache[group_id][i]) / difference_distance
                else:
                    break
            
            grouped_cache[group_id] = updated_taylor_factors
        # 新增
        cache_dic['grouped_cache'][layer_key][group_id] = grouped_cache[group_id]

def grouped_taylor_formula(cache_dic: Dict, current: Dict, feature_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    使用分组TaylorSeer计算预测结果
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息
        feature_shape: 特征形状 (N, D)
        
    Returns:
        predicted_feature: 预测的特征 [N, D]
    """
    layer_key = (current['model'], current['layer'], current['block'])
    grouper = cache_dic['dimension_grouper']
        
    if not grouper.is_fitted:
        # 如果还没有聚类，使用原始方法
        x = current['step'] - current['activated_steps'][-1]
        
        cache_data = cache_dic['cache'][-1][current['model']][current['layer']][current['block']]
        if len(cache_data) == 0:
            # 如果没有缓存数据，返回零tensor
            _, N, D = feature_shape
            # 尝试从grouped_cache获取正确的数据类型
            try:
                # 从grouped_cache中获取参考tensor
                for gid in cache_dic.get('grouped_cache', {}).get(layer_key, {}):
                    group_cache = cache_dic['grouped_cache'][layer_key][gid]
                    if len(group_cache) > 0:
                        ref_tensor = group_cache[0]
                        return torch.zeros((N, D), device=ref_tensor.device, dtype=ref_tensor.dtype)
            except:
                pass
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros((N, D), device=device, dtype=torch.float32)
        
        # 获取正确的数据类型和设备
        first_key = next(iter(cache_data.keys()))
        reference_tensor = cache_data[first_key]
        target_dtype = reference_tensor.dtype
        target_device = reference_tensor.device
        _, N, D = feature_shape
        
        output = None
        for i in range(len(cache_data)):
            import math
            term = (1 / math.factorial(i)) * cache_data[i] * (x ** i)
            if output is None:
                output = term
            else:
                output = output + term
        
        result = output if output is not None else torch.zeros((N, D), device=target_device, dtype=target_dtype)
        return result.to(dtype=target_dtype, device=target_device)
    
    # 使用分组方法预测
    dimension_groups = grouper.get_dimension_groups()
    grouped_cache = cache_dic['grouped_cache'][layer_key]
    
    x = current['step'] - current['activated_steps'][-1]
    
    # 初始化输出 - 处理不同的特征形状
    if len(feature_shape) == 3:
        batch_size, N, D = feature_shape  # shape: (batch, seq_len, feature_dim)
    elif len(feature_shape) == 2:
        N, D = feature_shape  # shape: (seq_len, feature_dim)
        batch_size = 1
    else:
        print(f"[GroupedTaylor] Error: Unsupported feature_shape {feature_shape}")
        return torch.zeros(feature_shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.bfloat16)
    
    # 安全地获取设备信息和数据类型
    try:
        cache_data = cache_dic['cache'][-1][current['model']][current['layer']][current['block']]
        if isinstance(cache_data, dict) and len(cache_data) > 0:
            # 从缓存中的第一个可用tensor获取设备和数据类型
            first_key = next(iter(cache_data.keys()))
            first_tensor = cache_data[first_key]
            device = first_tensor.device
            dtype = first_tensor.dtype
        else:
            # 尝试从grouped_cache获取参考信息
            for gid in grouped_cache:
                if len(grouped_cache[gid]) > 0:
                    ref_tensor = grouped_cache[gid][0]
                    device = ref_tensor.device
                    dtype = ref_tensor.dtype
                    break
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    except (KeyError, AttributeError, StopIteration):
        # 最后的fallback，尝试从grouped_cache获取信息
        try:
            for gid in grouped_cache:
                if len(grouped_cache[gid]) > 0:
                    ref_tensor = grouped_cache[gid][0]
                    device = ref_tensor.device
                    dtype = ref_tensor.dtype
                    break
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # 初始化输出张量 - 使用正确的形状
    if len(feature_shape) == 3:
        output = torch.zeros(feature_shape, device=device, dtype=dtype)
    else:
        output = torch.zeros((N, D), device=device, dtype=dtype)
    
    # 为每个组分别预测
    for group_id in range(grouper.n_clusters):
        group_mask = (dimension_groups == group_id)
        group_order = grouper.get_orders_for_groups()[group_id]
        
        if group_id in grouped_cache and len(grouped_cache[group_id]) > 0:
            # 检查是否使用 FoCa 算法 (order=4)
            if group_order == 4:
                # 使用 FoCa 算法
                from . import taylor_formula_foca
                
                # 为 FoCa 算法准备 cache_dic 和 current 参数
                foca_current = current.copy()
                foca_cache_dic = {
                    'cache': [{current['model']: {current['layer']: {current['block']: []}}}]
                }
                
                # 将组缓存转换为 FoCa 需要的格式（函数值序列）
                group_cache = grouped_cache[group_id]
                if len(group_cache) > 0:
                    # 假设 group_cache[0] 是最新值，group_cache[1] 是前一步等
                    feats_list = []
                    for i in range(min(4, len(group_cache))):  # 最多使用4个历史值
                        if i in group_cache:
                            feats_list.append(group_cache[i])
                    
                    foca_cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = feats_list
                    
                    try:
                        group_output = taylor_formula_foca(foca_cache_dic, foca_current)
                        if group_output is not None:
                            group_output = group_output.to(dtype=output.dtype)
                            if len(feature_shape) == 3:
                                output[:, :, group_mask] = group_output
                            else:
                                output[:, group_mask] = group_output
                    except Exception as e:
                        # 回退到传统方法
                        group_output = None
                        import math
                        for i in range(len(group_cache)):
                            term = (1 / math.factorial(i)) * group_cache[i] * (x ** i)
                            if group_output is None:
                                group_output = term
                            else:
                                group_output = group_output + term
                        
                        if group_output is not None:
                            group_output = group_output.to(dtype=output.dtype)
                            if len(feature_shape) == 3:
                                output[:, :, group_mask] = group_output
                            else:
                                output[:, group_mask] = group_output
            else:
                # 使用传统的 Taylor 展开方法
                group_output = None
                import math
                
                for i in range(len(grouped_cache[group_id])):
                    term = (1 / math.factorial(i)) * grouped_cache[group_id][i] * (x ** i)
                    if group_output is None:
                        group_output = term
                    else:
                        group_output = group_output + term
                
                if group_output is not None:
                    # 确保 group_output 与 output 具有相同的数据类型
                    group_output = group_output.to(dtype=output.dtype)
                    if len(feature_shape) == 3:
                        output[:, :, group_mask] = group_output
                    else:
                        output[:, group_mask] = group_output

    
    return output