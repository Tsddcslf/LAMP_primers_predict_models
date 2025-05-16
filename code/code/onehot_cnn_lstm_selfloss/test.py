import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

# 数据和模型路径
test_data_path = '../../data/data_lpb_2/test_data.csv'
model_path = 'best_model.pth'
result_path = 'result_onehot_self.csv'

# DNA序列的one-hot编码函数
def seq_to_onehot(seq):
    """
    将DNA序列转换为one-hot编码
    
    参数:
    seq -- str, DNA序列
    
    返回:
    onehot -- numpy array, one-hot编码的序列
    """
    # 定义碱基到索引的映射 (A:0, C:1, G:2, T:3)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()
    # 创建矩阵 [序列长度, 4] (4个通道对应A,C,G,T)
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            onehot[i, mapping[nucleotide]] = 1.0
        else:
            # 对于非ACGT的碱基(如N)，使用平均分布
            onehot[i] = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    return onehot

# 处理 enhanced_sequence 列
def process_enhanced_sequence(df):
    data_seq = df['enhanced_sequence'].values
    # 将序列转换为one-hot编码
    data_onehot = [seq_to_onehot(seq) for seq in data_seq]
    return data_onehot

# 处理 label_sequence 列
def process_label_sequence(df):
    data_labels_raw = df['label_sequence'].values
    # 将字符串标签转换为整数列表
    processed_labels = []
    for label_str in data_labels_raw:
        # 转换每个字符为对应的整数
        label_list = [int(digit) for digit in label_str]
        processed_labels.append(label_list)
    return processed_labels

# 定义模型结构 - 与训练文件中相同
class OneHotCNNBiLSTMClassifier(nn.Module):

    def __init__(self, dropout=0.3, hidden_size=256):
        super(OneHotCNNBiLSTMClassifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 简化CNN层 - 从one-hot编码的输入(4通道)开始
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(dropout)
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # MLP层
        self.linear1 = nn.Linear(hidden_size * 2, 256)  # BiLSTM输出是hidden_size*2
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 9)  # 9分类

    def forward(self, x):
        # x形状: [batch_size, seq_len, 4] (one-hot编码)
        
        # 调整维度以适应Conv1d [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        
        # 简化的CNN层，不使用池化
        x = F.relu(self.conv1(x))
        x = self.cnn_dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.cnn_dropout(x)
        
        # 调整维度以适应BiLSTM [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # BiLSTM层处理
        x, _ = self.bilstm(x)
        
        # MLP层处理
        x = self.dropout(x)
        x = self.relu1(self.linear1(x))
        x = self.dropout(x)
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)

        return x

# 测试数据集类
class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, max_length=500):
        self.labels = [label for label in df['labels']]
        self.sequences = [seq for seq in df['onehot']]
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 获取one-hot编码序列
        onehot_seq = self.sequences[index]
        
        # 处理序列长度
        seq_len = len(onehot_seq)
        if seq_len > self.max_length:
            # 截断
            onehot_seq = onehot_seq[:self.max_length]
        elif seq_len < self.max_length:
            # 填充
            padding = np.zeros((self.max_length - seq_len, 4), dtype=np.float32)
            onehot_seq = np.vstack([onehot_seq, padding])
        
        # 转换为张量
        onehot_tensor = torch.tensor(onehot_seq, dtype=torch.float32)
        
        # 获取标签
        label = self.labels[index]
        # 处理标签长度 - 注意：由于池化，标签长度需要调整
        if len(label) > self.max_length:
            label = label[:self.max_length]
        elif len(label) < self.max_length:
            label.extend([0] * (self.max_length - len(label)))
        
        # 确保标签是一个一维的张量
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return onehot_tensor, label_tensor

def main():
    print("开始测试模型...")
    
    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    
    # 处理测试数据
    test_onehot = process_enhanced_sequence(df_test)
    test_labels = process_label_sequence(df_test)
    df_test_processed = pd.DataFrame({'labels': test_labels, 'onehot': test_onehot})
    
    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(df_test_processed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载训练好的模型: {model_path}")
    model = OneHotCNNBiLSTMClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 用于存储结果的列表
    all_true_labels = []
    all_pred_labels = []
    
    # 预测
    print("开始预测...")
    correct_positions = 0
    total_positions = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=-1)
            
            # 计算正确预测的位置数量
            correct_mask = (preds == labels)
            correct_positions += torch.sum(correct_mask).item()
            total_positions += labels.numel()
            
            # 收集真实标签和预测标签
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())
    
    # 计算准确率
    accuracy = correct_positions / total_positions
    print(f"测试准确率: {accuracy:.4f}")
    
    # 准备结果数据
    print("处理结果...")
    result_data = []
    
    for true_seq, pred_seq in zip(all_true_labels, all_pred_labels):
        # 将整个序列保存为一行
        true_seq_str = ''.join(str(label) for label in true_seq)
        pred_seq_str = ''.join(str(label) for label in pred_seq)
        result_data.append({
            'true_label_sequence': true_seq_str,
            'predicted_label_sequence': pred_seq_str
        })
    
    # 创建结果DataFrame并保存
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(result_path, index=False)
    print(f"结果已保存到: {result_path}")
    
    # 分析每个类别的预测情况
    all_true_flat = np.array([label for sublist in all_true_labels for label in sublist])
    all_pred_flat = np.array([pred for sublist in all_pred_labels for pred in sublist])
    
    # 计算每个类别的准确率
    for cls in range(9):
        cls_mask = (all_true_flat == cls)
        if np.sum(cls_mask) > 0:
            cls_correct = np.sum((all_true_flat == all_pred_flat) & cls_mask)
            cls_total = np.sum(cls_mask)
            cls_acc = cls_correct / cls_total
            print(f"类别 {cls} 的准确率: {cls_acc:.4f} ({cls_correct}/{cls_total})")

if __name__ == "__main__":
    main() 