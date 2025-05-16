import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torchcrf import CRF  # 添加CRF库导入
import matplotlib.pyplot as plt

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

# 读取数据
train_data_path = '../../data/data_lpb_2/train_data.csv'
val_data_path = '../../data/data_lpb_2/val_data.csv'
test_data_path = '../../data/data_lpb_2/test_data.csv'

df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)

# 将输入序列转换成kmers格式用空格分开
def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

# 处理原始序列到kmers，处理原始标签，并将两者组合
k = 3  # k-mer 长度

def process_enhanced_sequence(df, k):
    data_seq = df['enhanced_sequence'].values
    data_kmer = [seq2kmer(seq, k) for seq in data_seq]
    return data_kmer

def process_label_sequence(df):
    data_labels_raw = df['label_sequence'].values
    processed_labels = []
    for label_str in data_labels_raw:
        label_list = [int(digit) for digit in label_str]
        processed_labels.append(label_list)
    return processed_labels

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
        self.linear1 = nn.Linear(hidden_size * 2, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 9)

        # CRF层
        self.crf = CRF(9, batch_first=True)

    def forward(self, input_id, mask, labels=None):
        output = self.bert(input_ids=input_id, attention_mask=mask)
        label_output = output[0]
        
        # CNN层
        x = label_output.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.cnn_dropout(x)
        x = F.relu(self.conv2(x))
        x = self.cnn_dropout(x)
        x = F.relu(self.conv3(x))
        x = self.cnn_dropout(x)
        
        x = x.permute(0, 2, 1)
        
        # BiLSTM层
        x, _ = self.bilstm(x)
        
        # MLP层
        x = self.dropout(x)
        x = self.relu1(self.linear1(x))
        x = self.dropout(x)
        x = self.relu2(self.linear2(x))
        emissions = self.linear3(x)
        
        crf_mask = mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
            return loss, emissions
        else:
            return self.crf.decode(emissions, mask=crf_mask), emissions

# 超参数设置
learning_rate = 0.00001
batch_size = 32
dropout_rate = 0.3
epochs = 100  # 继续训练的epochs数量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Dataset(df_train_processed)
val_dataset = Dataset(df_val_processed)
test_dataset = Dataset(df_test_processed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载已经训练好的模型
model = LAMPBertCNNBiLSTMClassifier(dropout=dropout_rate)
model.to(device)
model.load_state_dict(torch.load('final_model.pth'))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# 定义训练函数
def train_model(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        input_ids = inputs['input_ids'].squeeze(1).to(device)
        attention_mask = inputs['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, emissions = model(input_ids, attention_mask, labels)
        running_loss += loss.item()

        preds = model.crf.decode(emissions, mask=attention_mask.bool())
        batch_correct = 0
        batch_total = 0
        for i, pred_seq in enumerate(preds):
            pred_tensor = torch.tensor(pred_seq, device=device)
            seq_len = min(len(pred_seq), labels[i].size(0))
            batch_correct += torch.sum(pred_tensor[:seq_len] == labels[i][:seq_len])
            batch_total += seq_len
        
        correct_preds += batch_correct
        total_preds += batch_total

        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    return avg_loss, accuracy

# 验证函数
def validate_model(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            loss, emissions = model(input_ids, attention_mask, labels)
            running_loss += loss.item()

            preds = model.crf.decode(emissions, mask=attention_mask.bool())
            batch_correct = 0
            batch_total = 0
            for i, pred_seq in enumerate(preds):
                pred_tensor = torch.tensor(pred_seq, device=device)
                seq_len = min(len(pred_seq), labels[i].size(0))
                batch_correct += torch.sum(pred_tensor[:seq_len] == labels[i][:seq_len])
                batch_total += seq_len
            
            correct_preds += batch_correct
            total_preds += batch_total

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    return avg_loss, accuracy

# 训练继续
val_accuracies = []
for epoch in range(epochs):
    logger.info(f"Epoch {epoch + 1}/{epochs}")

    train_loss, train_acc = train_model(model, train_loader, optimizer, device)
    logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    val_loss, val_acc = validate_model(model, val_loader, device)
    val_accuracies.append(val_acc)
    logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

torch.save(model.state_dict(), 'final_model_continue.pth')

# 绘制验证准确率曲线
plt.figure(dpi=300, figsize=(10, 6))
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve (继续训练)')
plt.xticks(range(1, len(val_accuracies) + 1))
plt.legend()
plt.savefig('validation_accuracy_curve_continue.png')
plt.show()
