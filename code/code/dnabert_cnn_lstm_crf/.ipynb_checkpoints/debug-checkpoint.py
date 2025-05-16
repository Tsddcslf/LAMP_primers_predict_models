import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
import torch.nn as nn
import torch.nn.functional as F

# 参数
k = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = f'../../dnabertbased/{k}-mer'
model_path = 'best_model.pth'
test_csv = '../../data/data_lpb_2/test_data.csv'

# ===== 数据处理 =====
def seq2kmer(seq, k):
    return " ".join([seq[i:i+k] for i in range(len(seq)+1-k)])

def process_data(df):
    df['seq'] = df['enhanced_sequence'].apply(lambda x: seq2kmer(x, k))
    df['labels'] = df['label_sequence'].apply(lambda x: [int(c) for c in x])
    return df[['seq', 'labels']]

tokenizer = BertTokenizer.from_pretrained(bert_path)

class SimpleDataset(Dataset):
    def __init__(self, df):
        self.labels = df['labels'].tolist()
        self.texts = [tokenizer(s, padding='max_length', max_length=500, truncation=True, return_tensors='pt') for s in df['seq']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs, label

# ===== 模型结构（保持一致）=====
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
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = self.cnn_dropout(F.relu(self.conv1(x.permute(0, 2, 1))))
        x = self.cnn_dropout(F.relu(self.conv2(x)))
        x = self.cnn_dropout(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        emissions = self.linear3(x)
        return self.crf.decode(emissions, mask=attention_mask.bool())

# ===== 主流程 =====
def main():
    print("🧪 载入测试样本")
    df = pd.read_csv(test_csv)
    df = process_data(df)
    dataset = SimpleDataset(df)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)  # 用2个样本调试就够了

    print("📦 加载模型")
    model = LAMPBertCNNBiLSTMClassifier()
    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("🧩 missing keys:", missing)
    print("🧩 unexpected keys:", unexpected)

    model.to(device)
    model.eval()

    print("🚀 模型预测")
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            print("input_ids shape:", input_ids.shape)
            print("attention_mask shape:", attention_mask.shape)
            print("label shape:", labels.shape)

            preds = model(input_ids, attention_mask)

            for i, (true_seq, pred_seq) in enumerate(zip(labels, preds)):
                seq_len = min(len(true_seq), len(pred_seq))
                print(f"\nSample {i+1}")
                print("🔹 True   :", ''.join(str(x.item()) for x in true_seq[:seq_len]))
                print("🔹 Predict:", ''.join(str(x) for x in pred_seq[:seq_len]))
            break  # 只测一批就行

if __name__ == '__main__':
    main()
