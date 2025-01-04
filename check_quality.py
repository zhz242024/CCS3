import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 配置日志输出到标准输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def analyze_combined_data(processed_dir):
    """分析预处理后的联合特征数据质量"""
    MASK_VALUE = 0
    all_files = list(Path(processed_dir).glob("Participant*_features.npy"))
    
    # 初始化统计数据
    total_samples = 0
    total_masked = 0
    participant_stats = {}
    time_point_stats = None  # 用于记录每个时间点的有效数据比例
    feature_count = 18  # 确定特征数量为18
    dimension_stats = {i: {'valid': 0, 'masked': 0} for i in range(feature_count)}
    video_stats = {i: {'valid_ratio': []} for i in range(10)}  # 初始化视频统计

    logger.info(f"Found {len(all_files)} participant files")

    for file_path in tqdm(all_files, desc="Analyzing participants"):
        try:
            # 加载数据
            data = np.load(file_path)  # 期望形状 [10, 1000, 18]
            participant_id = file_path.stem.split('_')[0]
            
            if len(data.shape) != 3 or data.shape[2] != feature_count:
                logger.warning(f"{participant_id}: Unexpected shape {data.shape}")
                continue
            
            num_videos, num_timepoints, _ = data.shape

            # 初始化时间点统计矩阵
            if time_point_stats is None:
                time_point_stats = np.zeros((num_timepoints, feature_count))

            # 计算掩码比例
            valid_mask = (data != MASK_VALUE)  # [10, 1000, 18]
            total_samples += data.size
            total_masked += np.sum(~valid_mask)

            # 参与者级别统计
            valid_ratio = np.mean(valid_mask)
            participant_stats[participant_id] = {
                'valid_ratio': valid_ratio,
                'total_samples': data.size,
                'valid_samples': np.sum(valid_mask)
            }

            # 时间点统计 - 为每个特征分别统计
            time_point_stats += np.mean(valid_mask, axis=0)

            # 维度统计
            for dim in range(feature_count):
                dimension_stats[dim]['valid'] += np.sum(valid_mask[:, :, dim])
                dimension_stats[dim]['masked'] += np.sum(~valid_mask[:, :, dim])

            # 视频统计
            for video_idx in range(num_videos):
                video_valid_ratio = np.mean(valid_mask[video_idx])
                video_stats[video_idx]['valid_ratio'].append(video_valid_ratio)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")

    # 计算维度统计结果
    for dim in dimension_stats:
        total = dimension_stats[dim]['valid'] + dimension_stats[dim]['masked']
        dimension_stats[dim]['valid_ratio'] = dimension_stats[dim]['valid'] / total if total > 0 else 0

    # 计算视频统计结果
    for video_idx in video_stats:
        ratios = video_stats[video_idx]['valid_ratio']
        video_stats[video_idx]['mean_ratio'] = np.mean(ratios) if ratios else 0
        video_stats[video_idx]['std_ratio'] = np.std(ratios) if ratios else 0

    # 计算总体统计结果
    overall_stats = {
        'total_files': len(all_files),
        'total_samples': total_samples,
        'total_masked': total_masked,
        'overall_valid_ratio': 1 - (total_masked / total_samples) if total_samples > 0 else 0,
        'participant_mean_valid_ratio': np.mean([stats['valid_ratio'] for stats in participant_stats.values()]) if participant_stats else 0,
        'participant_std_valid_ratio': np.std([stats['valid_ratio'] for stats in participant_stats.values()]) if participant_stats else 0
    }
    
    # 时间点统计结果
    time_point_stats = time_point_stats / len(all_files) if len(all_files) > 0 else time_point_stats

    return {
        'overall': overall_stats,
        'participants': participant_stats,
        'time_points': time_point_stats,
        'dimensions': dimension_stats,
        'videos': video_stats,
        'feature_count': feature_count
    }

def plot_statistics(stats, output_dir):
    """绘制统计图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 检查参与者有效数据比例
    if 'participants' in stats and stats['participants']:
        plt.figure(figsize=(10, 6))
        valid_ratios = [min(max(stats['valid_ratio'], 0), 1) for stats in stats['participants'].values()]
        sns.histplot(valid_ratios, bins=10)
        plt.title('Distribution of Valid Data Ratio Across Participants')
        plt.xlabel('Valid Data Ratio')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'participant_distribution.png'))
        plt.close()
    else:
        logger.warning("No participant data for plotting.")

    # 2. 检查维度有效数据比例
    if 'dimensions' in stats and stats['dimensions']:
        plt.figure(figsize=(10, 6))
        dim_ratios = [min(max(stats['dimensions'][i]['valid_ratio'], 0), 1) for i in range(len(stats['dimensions']))]
        plt.bar([f'Dim {i+1}' for i in range(len(dim_ratios))], dim_ratios)
        plt.title('Valid Data Ratio by Dimension (Updated for 18 Features)')
        plt.ylabel('Valid Data Ratio')
        plt.savefig(os.path.join(output_dir, 'dimension_ratios.png'))
        plt.close()
    else:
        logger.warning("No dimension data for plotting.")

    # 3. 检查时间点有效数据比例
    if 'time_points' in stats and stats['time_points'].size > 0:
        plt.figure(figsize=(15, 5))
        for dim_idx in range(stats['time_points'].shape[1]):
            # 裁剪时间点的有效比例到 [0, 1]
            time_series = np.clip(stats['time_points'][:, dim_idx], 0, 1)
            plt.plot(time_series, label=f'Feature {dim_idx+1}')
        plt.title('Valid Data Ratio Along Time Points (Updated for 18 Features)')
        plt.xlabel('Time Point')
        plt.ylabel('Valid Data Ratio')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'time_series_quality.png'))
        plt.close()
    else:
        logger.warning("No time point data for plotting.")

    # 4. 检查视频有效数据比例
    if 'videos' in stats and stats['videos']:
        plt.figure(figsize=(12, 6))
        video_means = [min(max(stats['videos'][i]['mean_ratio'], 0), 1) for i in range(10)]
        video_stds = [min(stats['videos'][i]['std_ratio'], 1 - video_means[i]) for i in range(10)]
        plt.errorbar(range(10), video_means, yerr=video_stds, fmt='o', capsize=3)
        plt.title('Valid Data Ratio by Video')
        plt.xlabel('Video Index')
        plt.ylabel('Valid Data Ratio')
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(output_dir, 'video_comparison.png'))
        plt.close()
    else:
        logger.warning("No video data for plotting.")


def main():
    # 设置数据路径
    processed_dir = "/home/csd440/slurm_test/data/feature_all"
    output_dir = "/home/csd440/slurm_test/data/quality_analysis2"
    
    # 分析数据
    logger.info("Starting combined data quality analysis...")
    stats = analyze_combined_data(processed_dir)
    
    # 打印总体统计信息
    logger.info("\nOverall Statistics:")
    logger.info(f"Total files processed: {stats['overall']['total_files']}")
    logger.info(f"Overall valid data ratio: {stats['overall']['overall_valid_ratio']:.2%}")
    logger.info(f"Mean valid ratio across participants: {stats['overall']['participant_mean_valid_ratio']:.2%}")
    logger.info(f"Std of valid ratio across participants: {stats['overall']['participant_std_valid_ratio']:.2%}")
    
    # 生成可视化
    logger.info("\nGenerating visualizations...")
    plot_statistics(stats, output_dir)
    logger.info(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
