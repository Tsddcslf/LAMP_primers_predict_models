import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import logging
import torch.nn.functional as F
from typing import Optional

# 设置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 添加文件处理器
file_handler = logging.FileHandler('training_log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# 标签映射（0, 1, 3, 4, 5, 7, 8）->（0, 1, 2, 3, 4, 5, 6）
def map_labels(label_list):
    label_mapping = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6}
    return [label_mapping[label] for label in label_list]

# DNA序列的one-hot编码函数
def seq_to_onehot(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            onehot[i, mapping[nucleotide]] = 1.0
        else:
            onehot[i] = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    return onehot

# 处理 enhanced_sequence 列
def process_enhanced_sequence(df):
    data_seq = df['enhanced_sequence'].values
    data_onehot = [seq_to_onehot(seq) for seq in data_seq]
    return data_onehot

# 处理 label_sequence 列
def process_label_sequence(df):
    data_labels_raw = df['label_sequence'].values
    processed_labels = []
    for label_str in data_labels_raw:
        label_list = [int(digit) for digit in label_str]
        # 映射标签
        mapped_labels = map_labels(label_list)
        processed_labels.append(mapped_labels)
    return processed_labels

# 读取 k-fold 数据并进行处理
def load_kfold_data(fold):
    train_file = f'../../data/k_fold/train_{fold}_500.csv'
    val_file = f'../../data/k_fold/val_{fold}_500.csv'
    
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    
    # 处理数据
    train_onehot = process_enhanced_sequence(df_train)
    train_labels = process_label_sequence(df_train)
    val_onehot = process_enhanced_sequence(df_val)
    val_labels = process_label_sequence(df_val)
    
    # 返回处理后的数据
    return (train_onehot, train_labels), (val_onehot, val_labels)

# 创建 Dataset 类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, max_length=500):
        self.labels = labels
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        onehot_seq = self.sequences[index]
        
        # 处理序列长度
        seq_len = len(onehot_seq)
        if seq_len > self.max_length:
            onehot_seq = onehot_seq[:self.max_length]
        elif seq_len < self.max_length:
            padding = np.zeros((self.max_length - seq_len, 4), dtype=np.float32)
            onehot_seq = np.vstack([onehot_seq, padding])
        
        onehot_tensor = torch.tensor(onehot_seq, dtype=torch.float32)
        
        label = self.labels[index]
        if len(label) > self.max_length:
            label = label[:self.max_length]
        elif len(label) < self.max_length:
            label.extend([0] * (self.max_length - len(label)))
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return onehot_tensor, label_tensor

# 修改模型结构，使用one-hot编码输入
class OneHotCNNBiLSTMClassifier(nn.Module):
    def __init__(self, dropout=0.3, hidden_size=256):
        super(OneHotCNNBiLSTMClassifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        self.linear1 = nn.Linear(hidden_size * 2, 256)  # BiLSTM输出是hidden_size*2
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 7)  # 7分类

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整输入维度以适应Conv1d
        x = F.relu(self.conv1(x))
        x = self.cnn_dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.cnn_dropout(x)
        
        x = x.permute(0, 2, 1)  # 转换为 LSTM 所需的形状
        x, _ = self.bilstm(x)
        
        x = self.dropout(x)
        x = self.relu1(self.linear1(x))
        x = self.dropout(x)
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)
        
        return x

# 超参数设置
learning_rate = 0.0001
batch_size = 64
dropout_rate = 0.3
epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算类别权重
def compute_class_weights(train_labels):
    all_labels = [label for sublist in train_labels for label in sublist]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float32).to(device)

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算损失
        batch_size, seq_len, num_classes = outputs.size()
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1))  # 适配交叉熵损失
        running_loss += loss.item()

        # 计算精度
        _, preds = torch.max(outputs, dim=-1)
        correct_preds += torch.sum(preds == labels)
        total_preds += labels.numel()

        # 反向传播
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy

# 定义验证函数
def validate_model(model, val_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            batch_size, seq_len, num_classes = outputs.size()
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            running_loss += loss.item()

            # 计算精度
            _, preds = torch.max(outputs, dim=-1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.numel()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy

# k折交叉验证训练
fold_results = []
final_fold_accuracies = []  # 每个fold的最后一个epoch准确率
final_fold_losses = []      # 每个fold的最后一个epoch loss
for fold in range(1, 11):  # 10-fold
    logger.info(f"Training fold {fold}")
    
    # 加载每一折的数据
    (train_onehot, train_labels), (val_onehot, val_labels) = load_kfold_data(fold)
    
    # 计算类别权重
    class_weights = compute_class_weights(train_labels)
    
    # 创建DataLoader
    train_dataset = Dataset(train_onehot, train_labels)
    val_dataset = Dataset(val_onehot, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = OneHotCNNBiLSTMClassifier(dropout=dropout_rate, hidden_size=256)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 添加类别权重
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 训练和验证
    val_accuracies, val_losses = [], []
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    # 保存每个fold的最后一个epoch的指标
    final_fold_accuracies.append(val_accuracies[-1].detach().cpu().item() if isinstance(val_accuracies[-1], torch.Tensor) else val_accuracies[-1])
    final_fold_losses.append(val_losses[-1])

    fold_results.append((val_accuracies, val_losses))

# 绘制十折验证结果
avg_val_accuracies = np.mean(
    [[acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in result[0]] for result in fold_results],
    axis=0
)
avg_val_losses = np.mean(
    [result[1] for result in fold_results],
    axis=0
)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(range(1, epochs + 1), avg_val_accuracies, marker='o', label='Avg Validation Accuracy')
plt.plot(range(1, epochs + 1), avg_val_losses, marker='x', label='Avg Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Avg Validation Accuracy and Loss')
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.savefig('avg_validation_metrics.png')
plt.show()

# 绘制每个fold的最终准确率和loss
fold_ids = list(range(1, 11))

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(fold_ids, final_fold_accuracies, marker='o', label='Final Accuracy per Fold')
plt.plot(fold_ids, final_fold_losses, marker='x', label='Final Loss per Fold')
plt.xlabel('Fold')
plt.ylabel('Metric')
plt.title('Final Accuracy and Loss per Fold')
plt.xticks(fold_ids)
plt.legend()
plt.grid(True)
plt.savefig('final_accuracy_loss_per_fold.png')
plt.show()
