import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from tqdm import tqdm
from torchcrf import CRF  # 添加CRF库导入
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 数据和模型路径
test_data_path = '../../data/data_lpb_2/test_data.csv'
model_path = 'final_model.pth'
result_path = 'result_crf.csv'
k = 3  # k-mer 长度

# 将输入序列转换成kmers格式用空格分开
def seq2kmer(seq, k):
    """将原始序列转换为kmers格式"""
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

# 处理 enhanced_sequence 列
def process_enhanced_sequence(df, k):
    data_seq = df['enhanced_sequence'].values
    data_kmer = [seq2kmer(seq, k) for seq in data_seq]
    return data_kmer

# 处理 label_sequence 列
def process_label_sequence(df):
    data_labels_raw = df['label_sequence'].values
    processed_labels = []
    for label_str in data_labels_raw:
        label_list = [int(digit) for digit in label_str]
        processed_labels.append(label_list)
    return processed_labels

# 定义模型结构，与dnabert_cnn_lstm_crf.py中保持一致
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

# 数据集类
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
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

def main():
    print("开始测试模型...")
    
    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    
    # 处理测试数据
    test_kmer = process_enhanced_sequence(df_test, k)
    test_labels = process_label_sequence(df_test)
    df_test_processed = pd.DataFrame({'labels': test_labels, 'seq': test_kmer})
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(f'../../dnabertbased/{k}-mer')
    
    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(df_test_processed, tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载训练好的模型: {model_path}")
    model = LAMPBertCNNBiLSTMClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 用于存储结果的列表
    all_true_labels = []
    all_pred_labels = []
    
    # 预测
    print("开始预测...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, labels = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            
            # 前向传播 - 使用CRF解码
            preds, _ = model(input_ids, attention_mask)
            
            # 收集数据
            for i, (label_seq, pred_seq) in enumerate(zip(labels, preds)):
                # 确保长度兼容
                seq_len = min(len(pred_seq), label_seq.size(0))
                
                # 收集真实标签和预测标签
                all_true_labels.append(label_seq[:seq_len].cpu().tolist())
                all_pred_labels.append(pred_seq[:seq_len])
    
    # 准备结果数据
    print("处理结果...")
    result_data = []
    
    for true_seq, pred_seq in zip(all_true_labels, all_pred_labels):
        # 将整个序列标签转换为字符串
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
    
    # 计算准确率
    correct_positions = 0
    total_positions = 0
    
    for true_seq, pred_seq in zip(all_true_labels, all_pred_labels):
        for i in range(min(len(true_seq), len(pred_seq))):
            if true_seq[i] == pred_seq[i]:
                correct_positions += 1
            total_positions += 1
    
    accuracy = correct_positions / total_positions
    print(f"测试准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main() 