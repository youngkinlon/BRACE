#!/usr/bin/env python3
"""
FLUX.1-dev 分片文件合并脚本
将 transformer/ 目录下的分片文件合并为单个 flux1-dev.safetensors 文件
"""

import json
import os
import time
import gc
import psutil
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

def format_size(bytes_size):
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"

def get_memory_info():
    """获取当前内存使用情况"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'rss_gb': memory_info.rss / (1024**3),
            'available_gb': system_memory.available / (1024**3),
            'percent': system_memory.percent
        }
    except:
        return None

def clear_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_disk_space(output_file, required_gb):
    """检查磁盘空间是否足够"""
    output_dir = Path(output_file).parent
    try:
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        
        print(f"💾 磁盘可用空间: {free_gb:.1f}GB")
        print(f"📏 需要空间: {required_gb:.1f}GB")
        
        if free_gb < required_gb + 1:  # 额外留1GB缓冲
            print(f"⚠️  警告: 磁盘空间可能不足!")
            return False
        return True
    except:
        print("⚠️  无法检查磁盘空间")
        return True

def merge_flux_shards(model_dir: str, force: bool = False, resume: bool = True):
    """合并 FLUX 分片文件"""
    start_time = time.time()
    
    model_path = Path(model_dir)
    transformer_dir = model_path / "transformer"
    index_file = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    output_file = model_path / "flux1-dev.safetensors"
    temp_dir = model_path / ".merge_temp"
    progress_file = temp_dir / "merge_progress.json"
    
    print("🚀 FLUX.1-dev 分片合并脚本")
    print("=" * 50)
    print(f"📁 模型目录: {model_path}")
    print(f"📂 Transformer目录: {transformer_dir}")
    print(f"📋 索引文件: {index_file}")
    print(f"📤 输出文件: {output_file}")
    print("=" * 50)
    
    # 检查输出文件是否已存在
    if output_file.exists() and not force:
        print(f"✅ 发现已存在的合并文件: {output_file}")
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"📏 文件大小: {size_mb:.1f} MB")
        
        choice = input("\n是否重新合并? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("🔄 使用现有文件，跳过合并")
            return str(output_file)
    
    # 检查是否有未完成的合并进度
    completed_shards = set()
    if resume and progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_shards = set(progress_data.get('completed_shards', []))
                if completed_shards:
                    print(f"🔄 发现未完成的合并进度: {len(completed_shards)} 个分片已完成")
        except:
            print("⚠️  无法读取进度文件，将重新开始")
            completed_shards = set()
    
    # 检查必要文件
    if not index_file.exists():
        raise FileNotFoundError(f"❌ 索引文件不存在: {index_file}")
    
    if not transformer_dir.exists():
        raise FileNotFoundError(f"❌ Transformer目录不存在: {transformer_dir}")
    
    # 读取索引文件
    print("📖 读取索引文件...")
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data.get('weight_map', {})
    if not weight_map:
        raise ValueError("❌ 索引文件中没有找到 weight_map")
    
    # 获取所有分片文件
    shard_files = sorted(set(weight_map.values()))
    print(f"🔍 发现 {len(shard_files)} 个分片文件:")
    
    total_size = 0
    for shard in shard_files:
        shard_path = transformer_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"❌ 分片文件不存在: {shard_path}")
        
        size = shard_path.stat().st_size
        total_size += size
        print(f"  📦 {shard} - {format_size(size)}")
    
    print(f"📊 分片文件总大小: {format_size(total_size)}")
    
    # 检查磁盘空间
    required_gb = total_size / (1024**3)
    if not check_disk_space(output_file, required_gb):
        choice = input("\n⚠️  磁盘空间不足，是否继续? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("❌ 用户取消操作")
            return None
    
    print(f"\n🔄 开始合并 {len(weight_map)} 个权重参数...")
    
    # 显示初始内存状态
    mem_info = get_memory_info()
    if mem_info:
        print(f"💾 初始内存: {mem_info['rss_gb']:.1f}GB 使用, {mem_info['available_gb']:.1f}GB 可用")
    
    # 使用分批流式处理避免内存溢出
    merged_state_dict = {}
    batch_size = 100  # 每批处理权重数量
    total_weights = len(weight_map)
    
    # 按分片组织权重，减少重复加载
    shard_to_weights = {}
    for weight_name, shard_file in weight_map.items():
        if shard_file not in shard_to_weights:
            shard_to_weights[shard_file] = []
        shard_to_weights[shard_file].append(weight_name)
    
    # 创建临时目录用于保存进度
    temp_dir.mkdir(exist_ok=True)
    
    processed_weights = 0
    
    # 逐个分片处理
    for shard_idx, (shard_file, weight_names) in enumerate(shard_to_weights.items()):
        # 检查是否已经处理过这个分片
        if shard_file in completed_shards:
            print(f"  ⏭️  跳过已完成的分片: {shard_file}")
            processed_weights += len(weight_names)
            continue
            
        shard_path = transformer_dir / shard_file
        print(f"  📥 处理分片 {shard_idx+1}/{len(shard_to_weights)}: {shard_file}")
        
        # 显示内存状态
        mem_info = get_memory_info()
        if mem_info:
            print(f"      💾 内存状态: {mem_info['rss_gb']:.1f}GB 使用, {mem_info['available_gb']:.1f}GB 可用 ({mem_info['percent']:.1f}%)")
        
        shard_start = time.time()
        
        try:
            # 加载分片
            shard_data = load_file(str(shard_path))
            
            # 分批处理权重
            for i in range(0, len(weight_names), batch_size):
                batch_weights = weight_names[i:i+batch_size]
                
                for weight_name in batch_weights:
                    if weight_name in shard_data:
                        merged_state_dict[weight_name] = shard_data[weight_name]
                    else:
                        print(f"⚠️  警告: 权重 {weight_name} 在分片 {shard_file} 中未找到")
                    
                    processed_weights += 1
                
                # 定期显示进度和清理内存
                if processed_weights % 500 == 0:
                    progress = processed_weights / total_weights * 100
                    print(f"      📈 进度: {processed_weights}/{total_weights} ({progress:.1f}%)")
                    clear_memory()
            
            # 清理分片数据
            del shard_data
            clear_memory()
            
            # 记录已完成的分片
            completed_shards.add(shard_file)
            try:
                with open(progress_file, 'w') as f:
                    json.dump({'completed_shards': list(completed_shards)}, f)
            except:
                pass  # 忽略进度保存错误
            
            shard_elapsed = time.time() - shard_start
            print(f"      ✓ 完成 ({shard_elapsed:.1f}秒) - {len(weight_names)} 个权重")
            
        except Exception as e:
            print(f"      ❌ 加载分片失败: {e}")
            # 保存当前进度
            try:
                with open(progress_file, 'w') as f:
                    json.dump({'completed_shards': list(completed_shards)}, f)
                print(f"💾 已保存进度到: {progress_file}")
            except:
                pass
            raise
    
    print(f"\n✅ 权重合并完成! 总共 {len(merged_state_dict)} 个参数")
    
    # 显示最终内存状态
    mem_info = get_memory_info()
    if mem_info:
        print(f"💾 合并后内存: {mem_info['rss_gb']:.1f}GB 使用, {mem_info['available_gb']:.1f}GB 可用")
    
    # 计算合并后大小
    merged_size = sum(tensor.numel() * tensor.element_size() for tensor in merged_state_dict.values())
    print(f"📏 合并后大小: {format_size(merged_size)}")
    
    # 保存合并后的文件 - 分批保存以减少内存压力
    print(f"\n💾 保存合并文件到: {output_file}")
    save_start = time.time()
    
    # 检查内存情况，如果内存紧张就先清理
    mem_info = get_memory_info()
    if mem_info and mem_info['available_gb'] < 5:  # 可用内存少于5GB
        print("⚠️  内存紧张，执行垃圾回收...")
        clear_memory()
    
    try:
        save_file(merged_state_dict, str(output_file))
        save_elapsed = time.time() - save_start
        print(f"✅ 保存完成 ({save_elapsed:.1f}秒)")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        print("💡 建议: 尝试释放更多内存或使用更大内存的机器")
        raise
    
    # 验证输出文件
    if output_file.exists():
        output_size = output_file.stat().st_size
        print(f"📊 输出文件大小: {format_size(output_size)}")
        
        # 计算压缩比
        compression_ratio = (total_size - output_size) / total_size * 100
        print(f"📉 空间优化: {compression_ratio:.1f}% (节省 {format_size(total_size - output_size)})")
    
    # 清理临时文件
    try:
        if progress_file.exists():
            progress_file.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
        print("🧹 已清理临时文件")
    except:
        pass
    
    total_elapsed = time.time() - start_time
    print(f"\n🎉 合并完成! 总耗时: {total_elapsed:.1f}秒")
    print("=" * 50)
    
    return str(output_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='合并 FLUX.1-dev 分片文件为单个 safetensors 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python merge_shards.py resources/weights/FLUX.1-dev
  python merge_shards.py resources/weights/FLUX.1-dev --force
        """
    )
    
    parser.add_argument('model_dir', help='模型目录路径 (包含 transformer/ 子目录)')
    parser.add_argument('--force', action='store_true', help='强制重新合并，覆盖已存在的文件')
    parser.add_argument('--no-resume', action='store_true', help='不使用断点续传，重新开始合并')
    parser.add_argument('--check-only', action='store_true', help='仅检查文件状态，不执行合并')
    
    args = parser.parse_args()
    
    if args.check_only:
        # 仅检查模式
        model_path = Path(args.model_dir)
        transformer_dir = model_path / "transformer"
        output_file = model_path / "flux1-dev.safetensors"
        
        print("🔍 文件状态检查:")
        print(f"📁 模型目录: {model_path} - {'✅' if model_path.exists() else '❌'}")
        print(f"📂 Transformer目录: {'✅' if transformer_dir.exists() else '❌'}")
        print(f"📤 合并文件: {'✅' if output_file.exists() else '❌'}")
        
        if output_file.exists():
            size = format_size(output_file.stat().st_size)
            print(f"📏 合并文件大小: {size}")
        
        return 0
    
    try:
        output_file = merge_flux_shards(args.model_dir, args.force, resume=not args.no_resume)
        
        if output_file:
            print(f"\n🎯 成功! 合并后的文件: {output_file}")
            print(f"\n📝 现在可以在 sample.sh 中设置:")
            print(f'export FLUX_DEV="{output_file}"')
            print(f"\n🚀 或者直接运行测试:")
            print(f'bash sample.sh -m Taylor -i 1 -o 1 -l 1')
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断操作")
        return 130
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
