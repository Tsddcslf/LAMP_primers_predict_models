import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from transformers import BertModel, BertConfig, BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import logging
import torch.nn.functional as F
from typing import Optional
from torchcrf import CRF  # 添加CRF库导入

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

# 更新数据路径
train_data_path = '../../data/data_plus/train_data_500.csv'
val_data_path = '../../data/data_plus/val_data_500.csv'
test_data_path = '../../data/data_plus/test_data_500.csv'

# 读取数据
df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)

#将输入序列转换成kmers格式用空格分开
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

#处理原始序列到kmers，处理原始标签，并将两者组合
#可通过改变k的值，来切割不同kmers
k = 3  # k-mer 长度

# 处理 enhanced_sequence 列
def process_enhanced_sequence(df, k):
    data_seq = df['enhanced_sequence'].values
    data_kmer = [seq2kmer(seq, k) for seq in data_seq]
    return data_kmer

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

# 处理训练、验证和测试数据
train_kmer = process_enhanced_sequence(df_train, k)
train_labels = process_label_sequence(df_train)

val_kmer = process_enhanced_sequence(df_val, k)
val_labels = process_label_sequence(df_val)

test_kmer = process_enhanced_sequence(df_test, k)
test_labels = process_label_sequence(df_test)

# 创建 DataFrame
df_train_processed = pd.DataFrame({'labels': train_labels, 'seq': train_kmer})
df_val_processed = pd.DataFrame({'labels': val_labels, 'seq': val_kmer})
df_test_processed = pd.DataFrame({'labels': test_labels, 'seq': test_kmer})

tokenizer = BertTokenizer.from_pretrained(f'../../dnabertbased/{k}-mer')

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df):
        self.labels = [label for label in df['labels']]
        self.texts = [tokenizer(text, padding='max_length', max_length=500, truncation=True,
                                return_tensors="pt") for text in df['seq']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        inputs = self.texts[index]
        label = self.labels[index]
        # 确保标签是一个一维的张量
        label = torch.tensor(label, dtype=torch.long)
        return inputs, label


class LAMPBertCNNBiLSTMClassifier(nn.Module):

    def __init__(self, dropout=0.3, hidden_size=256):
        super(LAMPBertCNNBiLSTMClassifier, self).__init__()
        
        config = BertConfig.from_pretrained(f'../../dnabertbased/{k}-mer', output_attentions=True)
        self.bert = BertModel.from_pretrained(f'../../dnabertbased/{k}-mer', config=config)
        
        # 冻结BERT的所有层
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        # CNN层
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=hidden_size, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(dropout)  # 在CNN层后添加Dropout
        
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
        self.linear3 = nn.Linear(128, 9)  # 输出维度为9，作为CRF的输入
        
        # CRF层
        self.crf = CRF(9, batch_first=True)  # 9为标签数量

    def forward(self, input_id, mask, labels=None):
        # BERT输出
        output = self.bert(input_ids=input_id, attention_mask=mask)
        label_output = output[0]  # [batch_size, sequence_length, hidden_size]
        
        # CNN层
        x = label_output.permute(0, 2, 1)  # 调整维度以适应Conv1d [batch_size, hidden_size, sequence_length]
        x = F.relu(self.conv1(x))
        x = self.cnn_dropout(x)  # Dropout
        x = F.relu(self.conv2(x))
        x = self.cnn_dropout(x)  # Dropout
        x = F.relu(self.conv3(x))
        x = self.cnn_dropout(x)  # Dropout
        
        # 调整维度以适应BiLSTM [batch_size, sequence_length, hidden_size]
        x = x.permute(0, 2, 1)
        
        # BiLSTM层
        x, _ = self.bilstm(x)
        
        # MLP层
        x = self.dropout(x)
        x = self.relu1(self.linear1(x))
        x = self.dropout(x)
        x = self.relu2(self.linear2(x))
        emissions = self.linear3(x)  # 发射矩阵，CRF的输入
        
        # 创建mask，有效序列为True
        crf_mask = mask.bool()
        
        if labels is not None:
            # 训练模式，返回负对数似然损失
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
            return loss, emissions
        else:
            # 预测模式，返回最佳路径
            return self.crf.decode(emissions, mask=crf_mask), emissions

# 超参数设置
learning_rate = 0.00005
batch_size = 64
dropout_rate = 0.3
epochs = 200

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建DataLoader
train_dataset = Dataset(df_train_processed)
val_dataset = Dataset(df_val_processed)
test_dataset = Dataset(df_test_processed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 计算每个类别的样本数量并设置权重
all_labels_flat = [digit for sublist in train_labels for digit in sublist]
class_counts = np.bincount(all_labels_flat, minlength=9)  # 确保有9个类别
logger.info(f"类别分布: {class_counts}")

# 计算类别权重 (1/频率)
class_weights = np.zeros(9)
for i in range(9):
    if class_counts[i] > 0:
        class_weights[i] = 1.0 / class_counts[i]
    else:
        class_weights[i] = 0.0

# 归一化权重
class_weights = class_weights / class_weights.sum() * 9  # 乘以类别数，使权重和为类别数
logger.info(f"类别权重: {class_weights}")

# 将权重转换为张量
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 初始化模型、损失函数和优化器
model = LAMPBertCNNBiLSTMClassifier(dropout=dropout_rate)
model.to(device)
# 使用CRF自身的损失函数，不再需要交叉熵损失
# criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 使用统一的学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        input_ids = inputs['input_ids'].squeeze(1).to(device)  # 去除多余的维度并移到GPU
        attention_mask = inputs['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        loss, emissions = model(input_ids, attention_mask, labels)

        # CRF模型直接返回损失
        running_loss += loss.item()

        # 计算精度 - 使用CRF的decode方法获取预测
        preds = model.crf.decode(emissions, mask=attention_mask.bool())
        # 将预测列表转为tensor并计算准确率
        batch_correct = 0
        batch_total = 0
        for i, pred_seq in enumerate(preds):
            # 由于CRF decode返回的是列表，需要截断到实际序列长度
            pred_tensor = torch.tensor(pred_seq, device=device)
            # 计算每个批次中正确的预测
            seq_len = min(len(pred_seq), labels[i].size(0))  # 使用最小长度防止越界
            batch_correct += torch.sum(pred_tensor[:seq_len] == labels[i][:seq_len])
            batch_total += seq_len
        
        correct_preds += batch_correct
        total_preds += batch_total

        # 反向传播
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
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
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            # 前向传播
            loss, emissions = model(input_ids, attention_mask, labels)
            running_loss += loss.item()

            # 计算精度 - 使用CRF的decode方法获取预测
            preds = model.crf.decode(emissions, mask=attention_mask.bool())
            # 将预测列表转为tensor并计算准确率
            batch_correct = 0
            batch_total = 0
            for i, pred_seq in enumerate(preds):
                # 由于CRF decode返回的是列表，需要截断到实际序列长度
                pred_tensor = torch.tensor(pred_seq, device=device)
                # 计算每个批次中正确的预测
                seq_len = min(len(pred_seq), labels[i].size(0))  # 使用最小长度防止越界
                batch_correct += torch.sum(pred_tensor[:seq_len] == labels[i][:seq_len])
                batch_total += seq_len
            
            correct_preds += batch_correct
            total_preds += batch_total

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    return avg_loss, accuracy

# 测试模型并绘制混淆矩阵
def test_and_visualize(model, test_loader, device, class_names):
    model.eval()  # 设置为评估模式
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, labels = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            # 前向传播 - 使用预测模式
            preds, emissions = model(input_ids, attention_mask)  # 不传入labels使用预测模式
            
            # 处理预测结果和计算准确率
            batch_correct = 0
            batch_total = 0
            
            # 保存每个序列的标签和预测结果
            batch_labels = []
            batch_preds = []
            
            for i, pred_seq in enumerate(preds):
                # 由于CRF decode返回的是列表，需要转换并截断
                pred_tensor = torch.tensor(pred_seq, device=device)
                # 计算每个批次中正确的预测，使用最小长度防止越界
                seq_len = min(len(pred_seq), labels[i].size(0))
                batch_correct += torch.sum(pred_tensor[:seq_len] == labels[i][:seq_len])
                batch_total += seq_len
                
                # 为混淆矩阵收集标签和预测值
                batch_labels.append(labels[i][:seq_len].cpu().numpy())
                batch_preds.append(pred_tensor[:seq_len].cpu().numpy())
            
            correct_preds += batch_correct
            total_preds += batch_total
            
            all_labels.extend(batch_labels)
            all_preds.extend(batch_preds)

    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵
    all_labels_flat = [label for sublist in all_labels for label in sublist]
    all_preds_flat = [pred for sublist in all_preds for pred in sublist]
    cm = confusion_matrix(all_labels_flat, all_preds_flat)

    # 绘制并保存混淆矩阵
    def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
        plt.figure(figsize=(12, 8), dpi=100)
        np.set_printoptions(precision=2)
        ind_array = np.arange(len(class_names) + 1)
        x, y = np.meshgrid(ind_array, ind_array)
        diags = np.diag(cm)
        TP_FNs, TP_FPs = [], []

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            max_index = len(class_names)
            if x_val != max_index and y_val != max_index:
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, c, color='black', fontsize=15, va='center', ha='center')
            elif x_val == max_index and y_val != max_index:
                TP = diags[y_val]
                TP_FN = cm.sum(axis=1)[y_val]
                recall = TP / TP_FN if TP_FN > 0 else 0
                recall = f'{recall * 100:.2f}%' if recall > 0.01 else '0'
                TP_FNs.append(TP_FN)
                plt.text(x_val, y_val, f'{TP_FN}\n{recall}', color='black', va='center', ha='center')
            elif x_val != max_index and y_val == max_index:
                TP = diags[x_val]
                TP_FP = cm.sum(axis=0)[x_val]
                precision = TP / TP_FP if TP_FP > 0 else 0
                precision = f'{precision * 100:.2f}%' if precision > 0.01 else '0'
                TP_FPs.append(TP_FP)
                plt.text(x_val, y_val, f'{TP_FP}\n{precision}', color='black', va='center', ha='center')

        cm = np.insert(cm, max_index, TP_FNs, 1)
        cm = np.insert(cm, max_index, np.append(TP_FPs, sum(TP_FPs)), 0)
        plt.text(max_index, max_index, f'{sum(TP_FPs)}\n{accuracy * 100:.2f}%', color='red', va='center', ha='center')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(class_names)))
        plt.xticks(xlocations, class_names, rotation=45)
        plt.yticks(xlocations, class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_2.png')
        plt.show()

    plot_confusion_matrix(cm, class_names)

# 训练过程
best_val_acc = 0.0
val_accuracies = []  # 用于存储每个epoch的验证准确率

for epoch in range(epochs):
    logger.info(f"Epoch {epoch + 1}/{epochs}")

    # 训练
    train_loss, train_acc = train_model(model, train_loader, None, optimizer, device)
    logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # 验证
    val_loss, val_acc = validate_model(model, val_loader, None, device)
    val_accuracies.append(val_acc.cpu().numpy())  # 将张量移到CPU并转换为numpy
    logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# ✅ 保存最后一轮训练后的模型
torch.save(model.state_dict(), 'final_model.pth')
logger.info("模型已保存为 final_model.pth（最后一次训练结果）")
logger.info("Training complete.")

# 绘制验证准确率曲线并保存
plt.figure(dpi=300, figsize=(10, 6))
plt.plot(range(1, epochs + 1), val_accuracies, marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve')
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.savefig('validation_accuracy_curve_2.png')  # 保存图像
plt.show()

# 在训练完成后进行测试和可视化
class_names = ['NA', 'F1', 'LF', 'F2', 'F3', 'B1', 'LB', 'B2', 'B3']
print("Testing and visualizing...")
test_and_visualize(model, test_loader, device, class_names)