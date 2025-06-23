import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=11):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(11)

# 数据预处理类
class PowerDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_and_process_data(self, train_path, test_path):
        """加载并处理数据"""
        try:
            # 读取数据
            print(f"Loading data from {train_path} and {test_path}")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            print(f"Train data columns: {list(train_data.columns)}")
            
            # 显示数据类型
            print("\nData types in train data:")
            print(train_data.dtypes)
            
            # 合并数据以便统一处理
            all_data = pd.concat([train_data, test_data], ignore_index=True)
            
            # 数据处理
            processed_data = self.process_features(all_data)
            
            # 分割回训练和测试数据
            train_size = len(train_data)
            train_processed = processed_data[:train_size]
            test_processed = processed_data[train_size:]
            
            print(f"\nProcessed train data shape: {train_processed.shape}")
            print(f"Processed test data shape: {test_processed.shape}")
            
            return train_processed, test_processed
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check if the data files exist and have the correct format")
            raise
    
    def process_features(self, data):
        """特征处理"""
        # 复制数据
        processed = data.copy()
        
        # 定义数值列
        numeric_columns = [
            'Global_active_power', 'Global_reactive_power', 'Voltage', 
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
            'Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]
        
        # 转换数值列类型并处理缺失值
        for col in numeric_columns:
            if col in processed.columns:
                # 将非数值字符串转换为NaN
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
                # 用列的中位数填充缺失值
                if processed[col].isna().any():
                    median_value = processed[col].median()
                    processed[col].fillna(median_value, inplace=True)
                    print(f"Warning: {col} contains missing values, filled with median: {median_value}")
        
        # 计算sub_metering_remainder
        if ('Global_active_power' in processed.columns and 
            'Sub_metering_1' in processed.columns and
            'Sub_metering_2' in processed.columns and
            'Sub_metering_3' in processed.columns):
            
            # 确保所有列都是数值类型
            for col in ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
                if processed[col].dtype != np.number:
                    print(f"Warning: Converting {col} to numeric type")
                    processed[col] = pd.to_numeric(processed[col], errors='coerce')
            
            # 计算剩余计量
            processed['sub_metering_remainder'] = (
                processed['Global_active_power'] * 1000 / 60 - 
                processed['Sub_metering_1'] - 
                processed['Sub_metering_2'] - 
                processed['Sub_metering_3']
            )
        else:
            print("Warning: Missing columns for calculating sub_metering_remainder")
            processed['sub_metering_remainder'] = 0
        
        # 处理降水数据
        if 'RR' in processed.columns:
            processed['RR'] = processed['RR'] / 10
        
        # 添加时间特征
        if 'DateTime' in processed.columns:
            try:
                processed['DateTime'] = pd.to_datetime(processed['DateTime'])
                processed['month'] = processed['DateTime'].dt.month
                processed['day_of_year'] = processed['DateTime'].dt.dayofyear
                processed['weekday'] = processed['DateTime'].dt.weekday
            except:
                print("Warning: Could not parse DateTime column")
        
        # 检查是否还有非数值数据
        for col in numeric_columns + ['sub_metering_remainder']:
            if col in processed.columns:
                if not pd.api.types.is_numeric_dtype(processed[col]):
                    print(f"Warning: {col} is still not numeric after conversion")
        
        return processed
    
    def create_sequences(self, data, sequence_length, target_column='Global_active_power'):
        """创建时间序列数据"""
        # 选择特征列
        feature_columns = [
            'Global_active_power', 'Global_reactive_power', 'Voltage', 
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
            'Sub_metering_3', 'sub_metering_remainder', 'RR', 
            'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]
        
        # 添加时间特征（如果存在）
        if 'month' in data.columns:
            feature_columns.extend(['month', 'day_of_year', 'weekday'])
        
        # 确保所有特征列都存在且为数值类型
        available_features = []
        for col in feature_columns:
            if col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    available_features.append(col)
                else:
                    print(f"Warning: Skipping non-numeric column {col}")
        
        print(f"Using features: {available_features}")
        
        # 检查是否有足够的特征
        if len(available_features) == 0:
            raise ValueError("No valid numeric features found!")
        
        features = data[available_features].values
        target = data[target_column].values
        
        # 检查数据中是否还有NaN值
        if np.isnan(features).any():
            print("Warning: NaN values found in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0)
        
        if np.isnan(target).any():
            print("Warning: NaN values found in target, replacing with median")
            target_median = np.nanmedian(target)
            target = np.nan_to_num(target, nan=target_median)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length):
            X.append(features_scaled[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        
        return np.array(X), np.array(y)

# 数据集类
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            dim_feedforward=d_model * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# CNN-Transformer混合模型
class CNNTransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, 
                 cnn_channels=32, kernel_size=3, dropout=0.1):
        super(CNNTransformerModel, self).__init__()
        
        # 更复杂的CNN层
        self.conv1d_1 = nn.Conv1d(input_size, cnn_channels, kernel_size, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size, padding=1)
        self.relu_2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影层
        self.cnn_projection = nn.Linear(cnn_channels * 2, d_model)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 更复杂的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        
        # CNN特征提取
        x_cnn = x.transpose(1, 2)  # (batch, input_size, seq_len)
        cnn_features = self.relu_1(self.conv1d_1(x_cnn))
        cnn_features = self.relu_2(self.conv1d_2(cnn_features))
        cnn_features = cnn_features.transpose(1, 2)  # (batch, seq_len, cnn_channels)
        cnn_features = self.cnn_projection(cnn_features)
        
        # 原始特征投影
        x_proj = self.input_projection(x)
        
        # Transformer编码
        transformer_out = self.transformer(x_proj)
        cnn_transformer_out = self.transformer(cnn_features)
        
        # 特征融合
        combined = torch.cat([transformer_out.mean(dim=1), cnn_transformer_out.mean(dim=1)], dim=1)
        
        # 输出
        out = self.dropout(self.relu_fc(self.fc1(combined)))
        out = self.fc2(out)
        
        return out

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.002, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        if (epoch + 1) % 10 == 0:
            train_loss /= len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model, [], []

# 评估函数
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    # 计算标准差
    residuals = actuals - predictions
    std_residuals = np.std(residuals)
    
    return mse, mae, std_residuals, predictions, actuals

# 运行实验
def run_experiments(train_path, test_path, sequence_length=60, prediction_length=90, num_experiments=3):
    """运行实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据处理
    processor = PowerDataProcessor()
    train_data, test_data = processor.load_and_process_data(train_path, test_path)
    
    # 创建序列数据
    X_train, y_train = processor.create_sequences(train_data, sequence_length)
    X_test, y_test = processor.create_sequences(test_data, sequence_length)
    
    # 划分训练集和验证集
    split_idx = int(len(X_train) * 0.8)
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    input_size = X_train.shape[2]
    
    # 存储结果
    results = {
        'LSTM': {'mse': [], 'mae': [], 'std': []},
        'Transformer': {'mse': [], 'mae': [], 'std': []},
        'CNN-Transformer': {'mse': [], 'mae': [], 'std': []}
    }
    
    print(f"\n开始快速实验 - 序列长度: {sequence_length}, 预测长度: {prediction_length}")
    print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
    
    for exp in range(num_experiments):
        print(f"\n=== 实验 {exp+1}/{num_experiments} ===")
        set_seed(42 + exp)
        
        # 创建数据加载器
        train_dataset = PowerDataset(X_train_split, y_train_split)
        val_dataset = PowerDataset(X_val, y_val)
        test_dataset = PowerDataset(X_test, y_test)
        
        batch_size = min(64, len(train_dataset) // 4)  # 动态batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              pin_memory=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               pin_memory=True, num_workers=2)
        
        # 1. LSTM模型
        print("训练LSTM模型...")
        lstm_model = LSTMModel(input_size).to(device)
        lstm_model, _, _ = train_model(lstm_model, train_loader, val_loader, 
                                     num_epochs=30, device=device)
        mse, mae, std, _, _ = evaluate_model(lstm_model, test_loader, device)
        results['LSTM']['mse'].append(mse)
        results['LSTM']['mae'].append(mae)
        results['LSTM']['std'].append(std)
        print(f"LSTM - MSE: {mse:.6f}, MAE: {mae:.6f}, STD: {std:.6f}")
        
        # 2. Transformer模型
        print("训练Transformer模型...")
        transformer_model = TransformerModel(input_size).to(device)
        transformer_model, _, _ = train_model(transformer_model, train_loader, val_loader, 
                                            num_epochs=30, device=device)
        mse, mae, std, _, _ = evaluate_model(transformer_model, test_loader, device)
        results['Transformer']['mse'].append(mse)
        results['Transformer']['mae'].append(mae)
        results['Transformer']['std'].append(std)
        print(f"Transformer - MSE: {mse:.6f}, MAE: {mae:.6f}, STD: {std:.6f}")
        
        # 3. CNN-Transformer混合模型
        print("训练CNN-Transformer模型...")
        cnn_transformer_model = CNNTransformerModel(input_size).to(device)
        cnn_transformer_model, _, _ = train_model(cnn_transformer_model, train_loader, val_loader, 
                                                num_epochs=30, device=device)
        mse, mae, std, _, _ = evaluate_model(cnn_transformer_model, test_loader, device)
        results['CNN-Transformer']['mse'].append(mse)
        results['CNN-Transformer']['mae'].append(mae)
        results['CNN-Transformer']['std'].append(std)
        print(f"CNN-Transformer - MSE: {mse:.6f}, MAE: {mae:.6f}, STD: {std:.6f}")
    
    # 计算评价指标
    print(f"\n=== {prediction_length}天预测结果汇总 ===")
    summary_results = {}
    for model_name, metrics in results.items():
        mse_mean = np.mean(metrics['mse'])
        mse_std = np.std(metrics['mse'])
        mae_mean = np.mean(metrics['mae'])
        mae_std = np.std(metrics['mae'])
        std_mean = np.mean(metrics['std'])
        std_std = np.std(metrics['std'])
        
        print(f"{model_name}:")
        print(f"  MSE: {mse_mean:.6f} ± {mse_std:.6f}")
        print(f"  MAE: {mae_mean:.6f} ± {mae_std:.6f}")
        print(f"  STD: {std_mean:.6f} ± {std_std:.6f}")
        
        summary_results[model_name] = {
            'MSE_mean': mse_mean,
            'MSE_std': mse_std,
            'MAE_mean': mae_mean,
            'MAE_std': mae_std,
            'STD_mean': std_mean,
            'STD_std': std_std
        }
    
    return summary_results

if __name__ == "__main__":
    # 数据文件路径
    train_path = "path/to/your/train.csv"
    test_path = "path/to/your/train.csv"
    
    print("电力预测任务开始...")
    
    # 短期预测（90天）
    print("\n" + "="*50)
    print("短期预测（未来90天）")
    print("="*50)
    short_term_results = run_experiments(train_path, test_path, 
                                       sequence_length=60, prediction_length=90)
    
    # 长期预测（365天）
    print("\n" + "="*50)
    print("长期预测（未来365天）")
    print("="*50)
    long_term_results = run_experiments(train_path, test_path, 
                                      sequence_length=60, prediction_length=365)
    
    # 将结果保存为Excel文件
    short_term_df = pd.DataFrame(short_term_results).T
    long_term_df = pd.DataFrame(long_term_results).T
    
    with pd.ExcelWriter('evaluation_results.xlsx') as writer:
        short_term_df.to_excel(writer, sheet_name='Short_Term')
        long_term_df.to_excel(writer, sheet_name='Long_Term')
    
    print("\n实验完成！评估结果已保存为evaluation_results.xlsx")
