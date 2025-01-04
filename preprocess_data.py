import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MASK_VALUE = 0.0


def calculate_velocity(data):
    """计算速度特征"""
    x_diff = np.diff(data['gaze_point_3d_x'])
    y_diff = np.diff(data['gaze_point_3d_y'])
    z_diff = np.diff(data['gaze_point_3d_z'])
    time_diff = np.diff(data['gaze_timestamp'])
    
    velocity = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2) / (time_diff + 1e-7)
    return np.pad(velocity, (0, 1), mode='edge')

def calculate_amplitude(data):
    """计算幅度特征"""
    v1 = np.array([data['gaze_normal0_x'], data['gaze_normal0_y'], data['gaze_normal0_z']]).T
    v2 = np.array([data['gaze_normal1_x'], data['gaze_normal1_y'], data['gaze_normal1_z']]).T
    
    dot_product = np.sum(v1 * v2, axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    dot_product = np.clip(dot_product / (norms + 1e-7), -1, 1)
    return np.arccos(dot_product)

def calculate_ellipse_ratio(data):
    """计算椭圆轴比率（ellipse axis ratio）。"""
    axis_a = data['ellipse_axis_a']
    axis_b = data['ellipse_axis_b']
    
    # 防止分母为 0 的异常
    max_axis = np.maximum(axis_a, axis_b)
    min_axis = np.minimum(axis_a, axis_b)
    
    ratio = np.divide(min_axis, max_axis, out=np.zeros_like(min_axis), where=max_axis != 0)
    return ratio

def clean_data(df):
    """
    清理数据表：
    - 删除含 NaN 值的行。
    - 如果数据仍为空，返回空 DataFrame。
    """
    if df.empty:
        return df
    df = df.dropna()  # 删除含 NaN 值的行
    return df if not df.empty else pd.DataFrame()

def is_valid_excel(file_path):
    """
    检查 Excel 文件是否为有效的表格数据。
    - 文件必须可以被正确读取。
    - 表格行数和列数应合理。
    """
    try:
        df = pd.read_excel(file_path)
        # 简单检查：行数和列数必须满足基本要求（可根据你的数据调整阈值）
        if df.shape[0] > 2 and df.shape[1] > 2:
            return True
        else:
            return False
    except Exception:
        return False

def pad_or_trim_feature(feature, target_length, fill_value=MASK_VALUE):
    """
    将特征填充或裁剪为指定长度。
    
    参数:
    - feature (array-like): 输入特征数组，可以是列表或 NumPy 数组。
    - target_length (int): 目标长度。
    - fill_value (float): 用于填充的值，默认是 MASK_VALUE。
    
    返回:
    - np.ndarray: 长度为 target_length 的特征数组。
    """
    # 转为 NumPy 数组
    feature = np.asarray(feature)

    # 检查长度
    if len(feature) > target_length:
        # 裁剪到目标长度
        return feature[:target_length]
    elif len(feature) < target_length:
        # 填充到目标长度
        padding = np.full(target_length - len(feature), fill_value)
        return np.concatenate([feature, padding])
    return feature

def process_blinks(blink_data, max_length=9423):
    """
    处理 blinks 数据，提取与眨眼相关的特征。
    
    参数:
    - blink_data (DataFrame): 包含眨眼信息的表格。
    - max_length (int): 特征的目标长度。

    返回:
    - list: 各种统计特征。
    """
    if blink_data.empty:
        return [0, 0, 0, 0, 0]  # 若数据为空，返回默认特征

    try:
        # 计算眨眼次数
        num_blinks = len(blink_data)

        # 计算眨眼时长相关特征
        durations = blink_data['duration'].values
        avg_duration = durations.mean() if len(durations) > 0 else 0
        min_duration = durations.min() if len(durations) > 0 else 0
        max_duration = durations.max() if len(durations) > 0 else 0

        # 计算眨眼频率（假设总时长为最后一个眨眼的结束时间减去第一个眨眼的开始时间）
        total_time = blink_data['end_timestamp'].iloc[-1] - blink_data['start_timestamp'].iloc[0]
        blink_frequency = num_blinks / total_time if total_time > 0 else 0

        return [num_blinks, avg_duration, min_duration, max_duration, blink_frequency]
    except Exception as e:
        logger.error(f"Error processing blinks: {str(e)}")
        return [0, 0, 0, 0, 0]

def process_video_data(gaze_data, pupil_data, blink_data, annotation_data, max_length=9423):
    try:
        # 清理输入数据
        gaze_data = clean_data(gaze_data)
        pupil_data = clean_data(pupil_data)
        blink_data = clean_data(blink_data)

        logger.debug(f"Loaded gaze_data: {gaze_data.shape}, pupil_data: {pupil_data.shape}")
        logger.debug(f"Columns in gaze_data: {list(gaze_data.columns)}")
        logger.debug(f"Columns in pupil_data: {list(pupil_data.columns)}")
        
        if gaze_data.empty or pupil_data.empty:
            logger.warning("Empty gaze or pupil data after cleaning, returning masked array")
            return np.full((max_length, 18), MASK_VALUE)  # 更新特征维度

        # Gaze特征
        gaze_features = {
            'timestamp': pad_or_trim_feature(
                gaze_data['gaze_timestamp'] - gaze_data['gaze_timestamp'].iloc[0], max_length),
            'eye_center_displacement': pad_or_trim_feature(np.sqrt(
                (gaze_data['eye_center0_3d_x'] - gaze_data['eye_center1_3d_x'])**2 +
                (gaze_data['eye_center0_3d_y'] - gaze_data['eye_center1_3d_y'])**2 +
                (gaze_data['eye_center0_3d_z'] - gaze_data['eye_center1_3d_z'])**2), max_length),
            'norm_pos_x': pad_or_trim_feature(gaze_data['norm_pos_x'], max_length),
            'norm_pos_y': pad_or_trim_feature(gaze_data['norm_pos_y'], max_length),
            'gaze_velocity': pad_or_trim_feature(calculate_velocity(gaze_data), max_length),
            'gaze_amplitude': pad_or_trim_feature(calculate_amplitude(gaze_data), max_length)
        }


        # Pupil特征
        pupil_features = {
            'ellipse_ratio': pad_or_trim_feature(calculate_ellipse_ratio(pupil_data), max_length),
            'diameter': pad_or_trim_feature(pupil_data['diameter'], max_length),
            'diameter_3d': pad_or_trim_feature(pupil_data['diameter_3d'], max_length),
            'sphere_radius': pad_or_trim_feature(pupil_data['sphere_radius'], max_length),
            'circle_3d_radius': pad_or_trim_feature(pupil_data['circle_3d_radius'], max_length),
            'theta': pad_or_trim_feature(pupil_data['theta'], max_length),
            'phi': pad_or_trim_feature(pupil_data['phi'], max_length)
        }

        # Blink特征
        blink_features = process_blinks(blink_data)

        # 合并特征
        combined_features = {**gaze_features, **pupil_features}
        feature_array = np.array([combined_features[key] for key in combined_features.keys()]).T

        # 将 blink 特征拼接到最后
        blink_array = np.tile(blink_features, (max_length, 1))  # 重复到与时间序列对齐
        final_features = np.concatenate([feature_array, blink_array], axis=1)

        return final_features
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return np.full((max_length, 18), MASK_VALUE)  # 更新特征维度

def process_participant_data(file_path, max_length=9423):
    """处理参与者的所有视频数据"""
    try:
        # 验证 Gaze 文件
        if not is_valid_excel(f"{file_path}_Gaze.xlsx"):
            logger.warning(f"Invalid Gaze Excel file for {file_path}")
            return np.full((10, max_length, 13), MASK_VALUE)

        # 验证 Pupil 文件
        if not is_valid_excel(f"{file_path}_Pupil.xlsx"):
            logger.warning(f"Invalid Pupil Excel file for {file_path}")
            return np.full((10, max_length, 13), MASK_VALUE)

        # 加载数据
        gaze_dict = pd.read_excel(f"{file_path}_Gaze.xlsx", sheet_name=None)
        pupil_dict = pd.read_excel(f"{file_path}_Pupil.xlsx", sheet_name=None)
        blink_dict = pd.read_excel(f"{file_path}_Blinks.xlsx", sheet_name=None)
        annotation_dict = pd.read_excel(f"{file_path}_Annotation.xlsx", sheet_name=None)

        all_videos_features = []
        
        for video_idx in range(1, 11):
            video_name = f"Video{video_idx}"
            logger.info(f"Processing {video_name}")
            
            video_features = process_video_data(
                gaze_dict.get(video_name, pd.DataFrame()),
                pupil_dict.get(video_name, pd.DataFrame()),
                blink_dict.get(video_name, pd.DataFrame()),
                annotation_dict.get(video_name, pd.DataFrame()),
                max_length
            )
            all_videos_features.append(video_features)
        
        result = np.stack(all_videos_features)
        logger.info(f"Final data shape: {result.shape}")  # (10, 1000, 13)
        return result
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return np.full((10, max_length, 13), MASK_VALUE)


def main():
    data_dir = "/home/csd440/slurm_test/data"
    participants_dir = os.path.join(data_dir, "participants_data")
    output_dir = os.path.join(data_dir, "feature_all")
    os.makedirs(output_dir, exist_ok=True)
    
    participants = [p.split('_')[0] for p in os.listdir(participants_dir) if p.endswith("_Gaze.xlsx")]
    
    for participant_id in tqdm(participants):
        logger.info(f"\nProcessing {participant_id}")
        file_path = os.path.join(participants_dir, participant_id)
        
        try:
            features = process_participant_data(file_path)
            output_file = os.path.join(output_dir, f"{participant_id}_features.npy")
            np.save(output_file, features)
            logger.info(f"Saved features to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {participant_id}: {str(e)}")

if __name__ == "__main__":
    main()