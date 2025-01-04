import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, f1_score
from absl import app, flags
import numpy as np
import logging
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import StratifiedKFold

# 设置全局 FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('info_path', '/home/csd440/slurm_test/data/participants_info.csv', 'Path to participants_info.csv')
flags.DEFINE_string('data_dir', '/home/csd440/slurm_test/data/feature_all', 'Directory containing feature .npy files')
flags.DEFINE_string('output_dir', '/home/csd440/slurm_test/models', 'Directory to save models and logs')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training and validation')
flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension of RNN layers')
flags.DEFINE_integer('num_layers', 2, 'Number of RNN layers')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate for RNN and FC layers')
flags.DEFINE_integer('epochs', 200, 'Number of training epochs')
flags.DEFINE_float('lr', 1e-4, 'Learning rate for optimizer')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for optimizer')
flags.DEFINE_integer('n_splits', 5, 'Number of folds for cross-validation')
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility')

# 全局设置
MASK_VALUE = 0.0
FEATURE_DIM = 18
NUM_CLASSES = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_best_threshold(labels, probs, step=0.01):
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.0, 1.0, step):
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            elif isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                # 假设二分类，如果alpha是单个标量
                self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class FeatureDataset(Dataset):
    def __init__(self, data_dir, participant_ids, labels):
        self.data_dir = data_dir
        self.participant_ids = participant_ids
        self.labels = labels

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        participant_id = self.participant_ids[idx]
        file_path = os.path.join(self.data_dir, f"{participant_id}_features.npy")
        features = np.load(file_path)
        label = self.labels[idx]
        
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        return features, label

class MaskingRNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, num_layers, dropout, mask_value):
        super(MaskingRNN, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.mask_value = mask_value

        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        batch_size, num_videos, seq_length, feature_dim = x.size()
        x = x.view(batch_size * num_videos, seq_length, feature_dim)

        mask = (x != self.mask_value).any(dim=-1).float()
        x = x * mask.unsqueeze(-1)

        rnn_out, _ = self.rnn(x)

        mask_sum = mask.sum(dim=1, keepdim=True).clamp_min(1)
        rnn_out = (rnn_out * mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        logits = self.classifier(rnn_out)
        logits = logits.view(batch_size, num_videos, -1)
        logits = logits.mean(dim=1)

        return logits

def plot_step_curve(step_indices, step_values, title, ylabel, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(step_indices, step_values, label=title, color='blue')
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def augment_features(features):
    noise = torch.randn_like(features) * 0.05  # 高斯噪声
    scale = torch.rand_like(features) * 0.1 + 0.95  # 随机缩放
    features = features * scale + noise

    mask = torch.rand_like(features) > 0.1  # 随机屏蔽
    features = features * mask
    return features


def calculate_metrics(labels, preds):
    conf_matrix = confusion_matrix(labels, preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "confusion_matrix": {
            "tn": tn, "fp": fp,
            "fn": fn, "tp": tp
        }
    }

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """
    训练函数，包含步级别的监控
    """
    global_step = 0
    step_losses = []       # 记录每个log step的训练loss
    step_indices = []      # 对应的step序号
    step_accuracies = []   # 每个log step的准确率
    epoch_val_stats = []   # 每个epoch的验证指标
    log_steps = 5          # 每隔多少个batch记录一次
    
    best_f1_overall = 0.0
    patience = 20
    no_improve = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        all_labels = []
        all_preds = []
        
        for batch_i, (features, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            global_step += 1
            features = features.to(device)
            features = augment_features(features)
            labels = labels.view(-1).to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 添加到这里

            # 记录当前学习率
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Current Learning Rate: {current_lr}")

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # 每隔log_steps记录一次step级别的指标
            if global_step % log_steps == 0:
                avg_loss = loss.item()
                step_losses.append(avg_loss)
                step_indices.append(global_step)
                
                # 计算当前batch的准确率
                corrects = (preds.cpu() == labels.cpu()).sum().item()
                batch_acc = corrects / len(labels)
                step_accuracies.append(batch_acc)
                
                logger.info(f"Step {global_step}: Loss={avg_loss:.4f}, Acc={batch_acc:.4f}")
        
        # 计算整个epoch的训练指标
        train_metrics = calculate_metrics(all_labels, all_preds)
        train_loss = epoch_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validating"):
                features = features.to(device)
                labels = labels.view(-1).to(device)
                
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # 获取类别1的概率（假设类别1为正类）
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 转换为 NumPy 数组
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        
        # 寻找最佳阈值
        best_threshold, best_f1 = find_best_threshold(val_labels, val_probs)
        
        # 基于最佳阈值生成预测
        val_preds = (val_probs > best_threshold).astype(int)
        
        # 计算指标
        val_metrics = calculate_metrics(val_labels, val_preds)
        val_loss = val_loss / len(val_loader)
        
        # 记录验证指标
        epoch_val_stats.append({
            'epoch': epoch,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'best_threshold': best_threshold
        })
        
        # 详细的日志记录
        logger.info(f"\nEpoch {epoch}/{epochs} Summary:")
        logger.info(f"Training - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}")
        logger.info(f"Training - Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
        logger.info(f"Validation - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f} at Threshold: {best_threshold:.2f}")
        logger.info(f"Validation - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        logger.info("Confusion Matrix:")
        logger.info(f"TN: {val_metrics['confusion_matrix']['tn']}, FP: {val_metrics['confusion_matrix']['fp']}")
        logger.info(f"FN: {val_metrics['confusion_matrix']['fn']}, TP: {val_metrics['confusion_matrix']['tp']}")
        
        # early stopping检查
        if val_metrics["f1"] > best_f1_overall:
            best_f1_overall = val_metrics["f1"]
            save_path = os.path.join(FLAGS.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
            
    # 保存step级别的训练曲线
    plot_step_curve(
        step_indices, 
        step_losses, 
        "Training Loss per Step", 
        "Loss",
        os.path.join(FLAGS.output_dir, "step_loss_curve.png")
    )
    
    plot_step_curve(
        step_indices, 
        step_accuracies, 
        "Training Accuracy per Step", 
        "Accuracy",
        os.path.join(FLAGS.output_dir, "step_accuracy_curve.png")
    )
    
    return {
        'step_data': {
            'indices': step_indices,
            'losses': step_losses,
            'accuracies': step_accuracies
        },
        'epoch_val_stats': epoch_val_stats
    }


def get_labels_from_info(info_path, participant_ids):
    """
    扩展的标签定义: 如果 CESD >= 16 或者 (stateAnxiety >= 44 且 CESD > 10) 则标记为 1 (高风险)，
    否则标记为 0 (低风险)
    """
    labels = {}
    try:
        info_df = pd.read_csv(info_path)
        for participant_id in participant_ids:
            numeric_id = int(participant_id.replace("Participant", ""))
            row = info_df[info_df['ParticipantID'] == numeric_id]
            if row.empty:
                raise ValueError(f"No data found for {participant_id}")
            
            cesd_score = row.iloc[0]['cesd']
            state_score = row.iloc[0]['stateAnxiety']
            
            labels[participant_id] = 1 if (cesd_score >= 16 or 
                                         (state_score >= 48 and cesd_score >= 10)) else 0
    except Exception as e:
        print(f"Error generating labels: {e}")
        return None
    return labels

def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * (class_counts + 1e-6))  # 添加平滑
    return torch.FloatTensor(weights)


def plot_training_metrics(training_stats, save_path):
    metrics = ['loss', 'f1', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        train_vals = []  # 训练指标
        val_vals = []    # 验证指标
        epochs = []      # 记录epoch

        for stat in training_stats:
            if metric == 'loss':
                train_vals.append(stat.get('train_loss', 0))
                val_vals.append(stat.get('val_loss', 0))
            else:
                train_vals.append(stat.get('train_metrics', {}).get(metric, 0))
                val_vals.append(stat.get('val_metrics', {}).get(metric, 0))
            epochs.append(stat.get('epoch', 0))
        
        # 绘制训练和验证曲线
        ax.plot(epochs, train_vals, 'b-', label=f'Train {metric}')
        ax.plot(epochs, val_vals, 'r-', label=f'Val {metric}')
        
        # 设置图表标题和标签
        ax.set_title(f'{metric.upper()} over epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
        
        # 标注最佳点
        if val_vals:
            best_val_idx = np.argmax(val_vals) if metric != 'loss' else np.argmin(val_vals)
            best_val = val_vals[best_val_idx]
            ax.annotate(f'Best: {best_val:.4f}',
                        xy=(epochs[best_val_idx], best_val),
                        xytext=(10, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.suptitle('Training Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def analyze_feature_importance(model, feature_dim):
    """分析特征重要性"""
    # 获取GRU层的输入权重矩阵
    weight_ih = model.rnn.weight_ih_l0.abs().cpu().detach()  # 前向GRU的权重
    weight_ih_back = model.rnn.weight_ih_l0_reverse.abs().cpu().detach()  # 后向GRU的权重
    
    # 每个特征的权重是该特征连接到所有门的权重之和
    feature_importance = torch.zeros(feature_dim)
    for i in range(feature_dim):
        # 对双向GRU，我们取两个方向权重的平均
        importance_forward = weight_ih[:, i].mean().item()
        importance_backward = weight_ih_back[:, i].mean().item()
        feature_importance[i] = (importance_forward + importance_backward) / 2
    
    # 归一化特征重要性分数
    feature_importance = feature_importance / feature_importance.sum()
    
    return feature_importance.numpy()

def main(argv):
    set_seed(FLAGS.seed)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # 准备数据
    participant_ids = [f"Participant{i}" for i in range(1, 50) if i not in [13, 38]]
    labels_dict = get_labels_from_info(FLAGS.info_path, participant_ids)
    if labels_dict is None:
        raise ValueError("Failed to generate labels from participants_info.csv")
    
    all_labels = [labels_dict[pid] for pid in participant_ids]
    
    # 打印整体标签分布
    label_counts = np.bincount(all_labels)
    logger.info(f"Overall label distribution: {label_counts}")
    
    # 初始化K-fold
    kfold = StratifiedKFold(n_splits=FLAGS.n_splits, shuffle=True, random_state=FLAGS.seed)
    
    # 存储每个fold的结果
    fold_results = []
    all_feature_importance = []

    # K-fold交叉验证
    for fold, (train_idx, val_idx) in enumerate(kfold.split(participant_ids, all_labels)):
        logger.info(f"\nStarting fold {fold+1}/{FLAGS.n_splits}")
        
        # 准备当前fold的数据
        train_ids = [participant_ids[i] for i in train_idx]
        val_ids = [participant_ids[i] for i in val_idx]
        
        train_labels = [labels_dict[pid] for pid in train_ids]
        val_labels = [labels_dict[pid] for pid in val_ids]
        
        # 打印当前fold的标签分布
        train_label_counts = np.bincount(train_labels)
        val_label_counts = np.bincount(val_labels)
        logger.info(f"Fold {fold+1} - Train labels distribution: {train_label_counts}")
        logger.info(f"Fold {fold+1} - Val labels distribution: {val_label_counts}")
        
        train_dataset = FeatureDataset(FLAGS.data_dir, train_ids, train_labels)
        val_dataset = FeatureDataset(FLAGS.data_dir, val_ids, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = MaskingRNN(FEATURE_DIM, 
                          FLAGS.hidden_dim, 
                          NUM_CLASSES, 
                          FLAGS.num_layers, 
                          FLAGS.dropout, 
                          MASK_VALUE).to(device)
        
        class_weights = calculate_class_weights(train_labels).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)


        optimizer = optim.AdamW(
            model.parameters(), 
            lr=FLAGS.lr, 
            weight_decay=FLAGS.weight_decay,
            betas=(0.9, 0.999)
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=FLAGS.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # 训练当前fold
        fold_stats = train(model, 
                          train_loader,
                          val_loader,
                          criterion,
                          optimizer,
                          scheduler,
                          FLAGS.epochs,
                          device)
                
        # 训练完成后，分析特征重要性
        feature_importance = analyze_feature_importance(model, FEATURE_DIM)
        all_feature_importance.append(feature_importance)
        
        # 打印当前fold的特征重要性
        logger.info(f"\nFold {fold+1} Feature Importance:")
        for i, importance in enumerate(feature_importance):
            logger.info(f"Feature {i+1}: {importance:.4f}")
        
        # 保存每个fold的结果
        fold_results.append(fold_stats)
        
        # 为当前fold保存训练曲线
        plot_training_metrics(
            fold_stats['epoch_val_stats'],
            os.path.join(FLAGS.output_dir, f"training_metrics_fold{fold+1}.png")
        )
        
        # 计算当前fold的最佳F1值
        best_f1 = max(stat['val_metrics']['f1'] for stat in fold_stats['epoch_val_stats'])
        logger.info(f"Fold {fold+1} - Best validation F1: {best_f1:.4f}")
    
    # 计算并输出所有fold的平均表现
    avg_best_f1 = np.mean([
        max(fold['epoch_val_stats'][i]['val_metrics']['f1'] 
            for i in range(len(fold['epoch_val_stats'])))
        for fold in fold_results
    ])
    avg_feature_importance = np.mean(all_feature_importance, axis=0)
    std_feature_importance = np.std(all_feature_importance, axis=0)
    
    logger.info("\nAverage Feature Importance Across Folds:")
    for i, (avg_imp, std_imp) in enumerate(zip(avg_feature_importance, std_feature_importance)):
        logger.info(f"Feature {i+1}: {avg_imp:.4f} (±{std_imp:.4f})")
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, FEATURE_DIM + 1), avg_feature_importance)
    plt.errorbar(range(1, FEATURE_DIM + 1), avg_feature_importance, 
                yerr=std_feature_importance, fmt='none', color='black', capsize=5)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(FLAGS.output_dir, "feature_importance.png"))
    plt.close()
    
    std_best_f1 = np.std([
        max(fold['epoch_val_stats'][i]['val_metrics']['f1']
            for i in range(len(fold['epoch_val_stats'])))
        for fold in fold_results
    ])
    
    logger.info("\nCross-validation Results:")
    logger.info(f"Average Best F1: {avg_best_f1:.4f} (±{std_best_f1:.4f})")
    for fold_idx, fold_result in enumerate(fold_results):
        best_f1 = max(stat['val_metrics']['f1'] for stat in fold_result['epoch_val_stats'])
        logger.info(f"Fold {fold_idx+1} Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    app.run(main)
