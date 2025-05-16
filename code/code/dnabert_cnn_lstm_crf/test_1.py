import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
import torch.nn as nn
import torch.nn.functional as F

# 参数配置
k = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data_path = '../../data/data_lpb_2/test_data.csv'
bert_path = f'../../dnabertbased/{k}-mer'

# ====== 数据处理函数 ======
def seq2kmer(seq, k):
    return " ".join([seq[x:x+k] for x in range(len(seq)+1-k)])

def process_enhanced_sequence(df, k):
    return [seq2kmer(seq, k) for seq in df['enhanced_sequence'].values]

def process_label_sequence(df):
    return [[int(d) for d in label_str] for label_str in df['label_sequence'].values]

# ====== 数据集类 ======
tokenizer = BertTokenizer.from_pretrained(bert_path)

class TestDataset(Dataset):
    def __init__(self, df):
        self.labels = df['labels'].tolist()
        self.texts = [tokenizer(seq, padding='max_length', max_length=500, truncation=True, return_tensors='pt') for seq in df['seq']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs, label

# ====== 模型结构 ======
class LAMPBertCNNBiLSTMClassifier(nn.Module):
    def __init__(self, dropout=0.3, hidden_size=256):
        super().__init__()
        config = BertConfig.from_pretrained(bert_path, output_attentions=True)
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv1d(768, 384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(384, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, hidden_size, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                              num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(hidden_size * 2, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 9)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.crf = CRF(9, batch_first=True)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = bert_output.permute(0, 2, 1)
        x = self.cnn_dropout(F.relu(self.conv1(x)))
        x = self.cnn_dropout(F.relu(self.conv2(x)))
        x = self.cnn_dropout(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        emissions = self.linear3(x)
        return self.crf.decode(emissions, mask=attention_mask.bool())

# ====== 主流程 ======
def main():
    print("🔄 加载测试数据...")
    df_test = pd.read_csv(test_data_path)
    test_seq = process_enhanced_sequence(df_test, k)
    test_labels = process_label_sequence(df_test)
    df_test_processed = pd.DataFrame({'labels': test_labels, 'seq': test_seq})

    test_dataset = TestDataset(df_test_processed)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("✅ 加载模型...")
    model = LAMPBertCNNBiLSTMClassifier()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    print("🚀 开始预测...")
    all_trues = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            inputs, labels = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            preds = model(input_ids, attention_mask)

            for i, pred_seq in enumerate(preds):
                seq_len = min(len(pred_seq), labels[i].size(0))
                true_str = ''.join(str(x.item()) for x in labels[i][:seq_len])
                pred_str = ''.join(str(x) for x in pred_seq[:seq_len])
                all_trues.append(true_str)
                all_preds.append(pred_str)

    print("📁 保存至 test_predictions.csv ...")
    df_out = pd.DataFrame({
        "True_Label": all_trues,
        "Predicted_Label": all_preds
    })
    df_out.to_csv("test_predictions.csv", index=False, encoding="utf-8-sig")
    print("✅ 完成！测试结果已保存。")

if __name__ == "__main__":
    main()
