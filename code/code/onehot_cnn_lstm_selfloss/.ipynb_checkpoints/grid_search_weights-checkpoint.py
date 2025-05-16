import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os
import json
from train import (
    Dataset, 
    OneHotCNNBiLSTMClassifier, 
    WeightedPositionCrossEntropyLoss,
    process_enhanced_sequence, 
    process_label_sequence
)

# 设置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 添加文件处理器
file_handler = logging.FileHandler('grid_search_log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 更新数据路径
train_data_path = '../../data/data_lpb_2/train_data.csv'
val_data_path = '../../data/data_lpb_2/val_data.csv'

# 读取数据
df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)

# 处理训练和验证数据
train_onehot = process_enhanced_sequence(df_train)
train_labels = process_label_sequence(df_train)

val_onehot = process_enhanced_sequence(df_val)
val_labels = process_label_sequence(df_val)

# 创建 DataFrame
df_train_processed = pd.DataFrame({'labels': train_labels, 'onehot': train_onehot})
df_val_processed = pd.DataFrame({'labels': val_labels, 'onehot': val_onehot})

# 超参数设置
learning_rate = 0.0001
batch_size = 64
dropout_rate = 0.3
epochs = 50  # 减少轮数以加快网格搜索速度
hidden_size = 256

# 定义网格搜索的参数组合
correct_weights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # 添加原始权重1.0
incorrect_weights = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]  # 添加原始权重1.0

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 创建DataLoader
train_dataset = Dataset(df_train_processed)
val_dataset = Dataset(df_val_processed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# 定义训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = batch
        inputs = inputs.to(device)  # [batch_size, seq_len, 4]
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算损失
        batch_size, seq_len, num_classes = outputs.size()
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
        running_loss += loss.item()

        # 计算精度
        _, preds = torch.max(outputs, dim=-1)  # 获取每个位置的预测类别
        correct_mask = (preds == labels)  # 创建掩码，标识哪些位置预测正确
        
        correct_preds += torch.sum(correct_mask)  # 计算每个位置的正确预测
        total_preds += labels.numel()  # 计算总的预测位置数

        # 反向传播
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds
    
    return avg_loss, accuracy

# 定义验证函数
def validate(model, val_loader, criterion, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
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
            correct_mask = (preds == labels)
            
            correct_preds += torch.sum(correct_mask)
            total_preds += labels.numel()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_preds / total_preds
    
    return avg_loss, accuracy

# 创建结果目录
results_dir = "grid_search_results"
os.makedirs(results_dir, exist_ok=True)

# 执行网格搜索
results = []
best_val_acc = 0.0
best_params = None

for correct_weight in correct_weights:
    for incorrect_weight in incorrect_weights:
        # 设置随机种子以确保公平比较
        set_seed(42)
        
        logger.info(f"开始训练模型，正确权重: {correct_weight}, 错误权重: {incorrect_weight}")
        
        # 初始化模型
        model = OneHotCNNBiLSTMClassifier(dropout=dropout_rate, hidden_size=hidden_size)
        model.to(device)
        
        # 使用自定义的带位置权重的损失函数
        criterion = WeightedPositionCrossEntropyLoss(
            weight=class_weights_tensor, 
            correct_weight=correct_weight, 
            incorrect_weight=incorrect_weight
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # 训练记录
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # 训练循环
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc.item())
            
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc.item())
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 记录最后的验证准确率
        final_val_acc = val_accs[-1]
        
        # 保存结果
        result = {
            "correct_weight": correct_weight,
            "incorrect_weight": incorrect_weight,
            "final_train_acc": train_accs[-1],
            "final_val_acc": final_val_acc,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "train_acc_history": train_accs,
            "val_acc_history": val_accs,
            "train_loss_history": train_losses,
            "val_loss_history": val_losses
        }
        
        results.append(result)
        
        # 保存结果到JSON文件
        result_filename = f"weight_c{correct_weight}_i{incorrect_weight}.json"
        with open(os.path.join(results_dir, result_filename), 'w') as f:
            json.dump(result, f, indent=4)
        
        # 更新最佳参数
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_params = (correct_weight, incorrect_weight)
            
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
        
        logger.info(f"完成当前参数组合训练。最终验证准确率: {final_val_acc:.4f}")
        logger.info("-" * 50)

# 保存所有结果到单个文件
with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# 输出最佳参数
logger.info(f"网格搜索完成！")
logger.info(f"最佳参数 - 正确权重: {best_params[0]}, 错误权重: {best_params[1]}")
logger.info(f"最佳验证准确率: {best_val_acc:.4f}")

# 可视化结果
def plot_heatmap(results, metric='final_val_acc'):
    """绘制热力图展示不同权重组合的性能"""
    # 提取参数和指标值
    c_weights = sorted(list(set([r['correct_weight'] for r in results])))
    i_weights = sorted(list(set([r['incorrect_weight'] for r in results])))
    
    # 创建网格
    grid = np.zeros((len(c_weights), len(i_weights)))
    
    # 填充网格
    for r in results:
        c_idx = c_weights.index(r['correct_weight'])
        i_idx = i_weights.index(r['incorrect_weight'])
        grid[c_idx, i_idx] = r[metric]
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='YlGnBu')
    
    # 添加数值标签
    for i in range(len(c_weights)):
        for j in range(len(i_weights)):
            plt.text(j, i, f"{grid[i, j]:.4f}", 
                     ha="center", va="center", color="black")
    
    # 设置轴标签
    plt.xticks(range(len(i_weights)), i_weights)
    plt.yticks(range(len(c_weights)), c_weights)
    plt.xlabel('Incorrect Weight')
    plt.ylabel('Correct Weight')
    
    # 设置标题
    title_map = {
        'final_val_acc': '最终验证准确率',
        'final_train_acc': '最终训练准确率',
        'final_val_loss': '最终验证损失',
        'final_train_loss': '最终训练损失'
    }
    plt.title(title_map.get(metric, metric))
    
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'heatmap_{metric}.png'))
    plt.close()

# 绘制各种指标的热力图
plot_heatmap(results, 'final_val_acc')
plot_heatmap(results, 'final_train_acc')
plot_heatmap(results, 'final_val_loss')
plot_heatmap(results, 'final_train_loss')

# 绘制最佳参数组合的训练曲线
def plot_best_learning_curves():
    """绘制最佳参数组合的学习曲线"""
    best_result = None
    for r in results:
        if r['correct_weight'] == best_params[0] and r['incorrect_weight'] == best_params[1]:
            best_result = r
            break
    
    if best_result:
        plt.figure(figsize=(12, 5))
        
        # 绘制准确率
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), best_result['train_acc_history'], 'b-', label='训练准确率')
        plt.plot(range(1, epochs + 1), best_result['val_acc_history'], 'r-', label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.title(f'最佳参数组合的准确率曲线\nCorrect: {best_params[0]}, Incorrect: {best_params[1]}')
        plt.legend()
        plt.grid(True)
        
        # 绘制损失
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), best_result['train_loss_history'], 'b-', label='训练损失')
        plt.plot(range(1, epochs + 1), best_result['val_loss_history'], 'r-', label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('最佳参数组合的损失曲线')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'best_learning_curves.png'))
        plt.close()

plot_best_learning_curves()

# 生成综合报告
def generate_report():
    """生成网格搜索结果的综合报告"""
    # 对结果按验证准确率排序
    sorted_results = sorted(results, key=lambda x: x['final_val_acc'], reverse=True)
    
    # 创建报告文件
    with open(os.path.join(results_dir, 'grid_search_report.txt'), 'w') as f:
        f.write("网格搜索结果报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"最佳参数组合:\n")
        f.write(f"  正确预测权重: {best_params[0]}\n")
        f.write(f"  错误预测权重: {best_params[1]}\n")
        f.write(f"  最终验证准确率: {best_val_acc:.4f}\n\n")
        
        f.write("所有参数组合结果（按验证准确率排序）:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'正确权重':^10} | {'错误权重':^10} | {'验证准确率':^10} | {'训练准确率':^10} | {'验证损失':^10} | {'训练损失':^10}\n")
        f.write("-" * 70 + "\n")
        
        for r in sorted_results:
            f.write(f"{r['correct_weight']:^10.2f} | {r['incorrect_weight']:^10.2f} | "
                   f"{r['final_val_acc']:^10.4f} | {r['final_train_acc']:^10.4f} | "
                   f"{r['final_val_loss']:^10.4f} | {r['final_train_loss']:^10.4f}\n")
        
        f.write("\n\n")
        f.write("结论与分析:\n")
        f.write("-" * 50 + "\n")
        f.write("1. 根据网格搜索结果，最优的权重组合为:\n")
        f.write(f"   - 正确预测权重: {best_params[0]}\n")
        f.write(f"   - 错误预测权重: {best_params[1]}\n\n")
        
        # 分析不同参数对结果的影响
        f.write("2. 正确预测权重对模型性能的影响:\n")
        for c_weight in correct_weights:
            subset = [r for r in results if r['correct_weight'] == c_weight]
            avg_val_acc = sum(r['final_val_acc'] for r in subset) / len(subset)
            f.write(f"   - 权重 {c_weight}: 平均验证准确率 {avg_val_acc:.4f}\n")
        
        f.write("\n3. 错误预测权重对模型性能的影响:\n")
        for i_weight in incorrect_weights:
            subset = [r for r in results if r['incorrect_weight'] == i_weight]
            avg_val_acc = sum(r['final_val_acc'] for r in subset) / len(subset)
            f.write(f"   - 权重 {i_weight}: 平均验证准确率 {avg_val_acc:.4f}\n")
        
        f.write("\n4. 总体趋势分析:\n")
        f.write("   (此处根据热力图和结果综合分析得出总体趋势)\n\n")
        
        f.write("5. 建议:\n")
        f.write(f"   - 在实际应用中，建议使用正确预测权重 {best_params[0]} 和错误预测权重 {best_params[1]}\n")
        f.write("   - 如果需要更精细的调整，可以在最佳参数附近进行更细粒度的搜索\n")

generate_report()

print("网格搜索完成！结果已保存到", results_dir, "目录") 