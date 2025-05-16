import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 数据和模型路径
test_data_path = '../../data/data_lpb_2/test_data.csv'
model_path = 'best_transformer_model.pth'
result_path = 'result_transformer.csv'
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

# 定义Transformer模型结构
class LAMPBertTransformerClassifier(nn.Module):
    def __init__(self, dropout=0.3, hidden_size=256):
        super(LAMPBertTransformerClassifier, self).__init__()
        
        config = BertConfig.from_pretrained(f'../../dnabertbased/{k}-mer', output_attentions=True)
        self.bert = BertModel.from_pretrained(f'../../dnabertbased/{k}-mer', config=config)
        
        # 冻结BERT的所有层
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,  # BERT输出的维度
            nhead=8,      # 多头注意力头数
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # MLP层
        self.linear1 = nn.Linear(768, 256)  # Transformer输出是768
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 9)  # 9分类

    def forward(self, input_id, mask):
        # BERT输出
        output = self.bert(input_ids=input_id, attention_mask=mask)
        x = output[0]  # [batch_size, sequence_length, hidden_size]
        
        # Transformer自注意力层
        # 使用BERT的attention_mask
        x = self.transformer_encoder(x, src_key_padding_mask=(1 - mask).bool())
        
        # MLP层
        x = self.dropout(x)
        x = self.relu1(self.linear1(x))
        x = self.dropout(x)
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)

        return x

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

def plot_confusion_matrix(cm, class_names, title='混淆矩阵', accuracy=0):
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
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('transformer_confusion_matrix.png')
    plt.show()

def main():
    print("开始测试Transformer模型...")
    
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
    model = LAMPBertTransformerClassifier()
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
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=-1)
            
            # 收集数据
            for i, (label_seq, pred_seq) in enumerate(zip(labels, preds)):
                # 只考虑有效的序列部分（非填充部分）
                mask = attention_mask[i]
                seq_len = mask.sum().item()
                
                # 收集真实标签和预测标签
                all_true_labels.extend(label_seq[:seq_len].cpu().tolist())
                all_pred_labels.extend(pred_seq[:seq_len].cpu().tolist())
    
    # 准备结果数据
    print("处理结果...")
    result_data = []
    
    for i in range(len(df_test)):
        true_seq = test_labels[i]
        # 预测标签可能长度不一，需要处理
        pred_seq_indices = []
        start_idx = 0
        
        for j in range(i):
            start_idx += len(test_labels[j])
        
        end_idx = start_idx + len(true_seq)
        pred_seq = all_pred_labels[start_idx:end_idx] if start_idx < len(all_pred_labels) else []
        
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
    correct_positions = sum(t == p for t, p in zip(all_true_labels, all_pred_labels))
    total_positions = len(all_true_labels)
    
    accuracy = correct_positions / total_positions if total_positions > 0 else 0
    print(f"测试准确率: {accuracy:.4f}")
    
    # 生成混淆矩阵
    print("生成混淆矩阵...")
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=range(9))
    class_names = ['NA', 'F1', 'LF', 'F2', 'F3', 'B1', 'LB', 'B2', 'B3']
    plot_confusion_matrix(cm, class_names, title='Transformer模型混淆矩阵', accuracy=accuracy)

if __name__ == "__main__":
    main()