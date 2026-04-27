import os
import random
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore")


# =========================================================
# 1. 参数设置
# =========================================================

TRAIN_PATH = "./LSTM-Multivariate_pollution.csv"
TEST_PATH = "./pollution_test_data1.csv"

LOOKBACK = 24          # 使用过去 24 小时预测下一小时
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

VAL_RATIO_IN_TRAIN = 0.1
SEED = 42


# =========================================================
# 2. 固定随机种子
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. 构造滑动窗口序列
# =========================================================

def build_sequences(X, y, lookback, start_target_idx, end_target_idx):
    """
    对于目标时刻 t：
    输入：X[t-lookback : t]
    标签：y[t]

    即：用过去 lookback 小时的污染和天气信息预测下一小时污染。
    """
    X_seq = []
    y_seq = []

    for t in range(start_target_idx, end_target_idx):
        start = t - lookback

        if start < 0:
            continue

        X_seq.append(X[start:t])
        y_seq.append(y[t])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# =========================================================
# 4. Dataset
# =========================================================

class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# 5. LSTM 模型
# =========================================================

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, lookback, input_size]
        """
        out, _ = self.lstm(x)

        # 取最后一个时间步的隐藏状态
        last_hidden = out[:, -1, :]

        pred = self.regressor(last_hidden)
        return pred


# =========================================================
# 6. 训练一个 epoch
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # 防止 LSTM 梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


# =========================================================
# 7. 验证 / 测试
# =========================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds.append(pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.vstack(preds)
    trues = np.vstack(trues)

    return avg_loss, preds, trues


# =========================================================
# 8. 主函数
# =========================================================

def main():
    print("程序开始运行")

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # -----------------------------------------------------
    # 8.1 读取数据
    # -----------------------------------------------------

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"找不到测试文件：{TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("原始大表形状：", train_df.shape)
    print("测试集形状：", test_df.shape)

    # -----------------------------------------------------
    # 8.2 避免数据泄漏
    # -----------------------------------------------------
    # 你发现得很对：pollution_test_data1.csv 实际上是大表最后一段。
    # 所以训练时必须从大表中删掉最后 len(test_df) 行。

    test_len = len(test_df)
    train_df = train_df.iloc[:-test_len].copy()

    print("去除测试段后的真实训练集形状：", train_df.shape)

    # -----------------------------------------------------
    # 8.3 处理 date
    # -----------------------------------------------------
    # date 用来保证时间顺序，但不直接作为模型输入特征。

    if "date" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"])#将date列转换为日期时间格式
        train_df = train_df.sort_values("date").reset_index(drop=True)#按日期排序并重置索引
        train_df = train_df.drop(columns=["date"])#删除date列

    if "date" in test_df.columns:
        test_df["date"] = pd.to_datetime(test_df["date"])
        test_df = test_df.sort_values("date").reset_index(drop=True)
        test_df = test_df.drop(columns=["date"])

    print("去掉 date 后训练集列名：", train_df.columns.tolist())
    print("去掉 date 后测试集列名：", test_df.columns.tolist())

    # -----------------------------------------------------
    # 8.4 缺失值处理
    # -----------------------------------------------------

    numeric_cols = [
        "pollution",
        "dew",
        "temp",
        "press",
        "wnd_spd",
        "snow",
        "rain"
    ]

    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    train_df[numeric_cols] = train_df[numeric_cols].ffill().bfill()
    test_df[numeric_cols] = test_df[numeric_cols].ffill().bfill()

    train_df["wnd_dir"] = train_df["wnd_dir"].astype(str)
    test_df["wnd_dir"] = test_df["wnd_dir"].astype(str)

    # -----------------------------------------------------
    # 8.5 风向 one-hot 编码
    # -----------------------------------------------------

    train_df = pd.get_dummies(train_df, columns=["wnd_dir"], drop_first=False)
    test_df = pd.get_dummies(test_df, columns=["wnd_dir"], drop_first=False)

    # 保证测试集列和训练集列完全一致
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    print("one-hot 后训练集形状：", train_df.shape)
    print("one-hot 后测试集形状：", test_df.shape)
    print("最终输入字段：", train_df.columns.tolist())

    # -----------------------------------------------------
    # 8.6 标准化
    # -----------------------------------------------------
    # 注意：只能用训练集 fit scaler，不能用测试集 fit，避免数据泄漏。

    feature_cols = train_df.columns.tolist()
    target_col = "pollution"

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[feature_cols])
    y_scaler.fit(train_df[[target_col]])

    X_train_all_raw = x_scaler.transform(train_df[feature_cols])
    y_train_all_raw = y_scaler.transform(train_df[[target_col]]).reshape(-1)

    # -----------------------------------------------------
    # 8.7 构造训练序列
    # -----------------------------------------------------

    X_train_all, y_train_all = build_sequences(
        X_train_all_raw,
        y_train_all_raw,
        lookback=LOOKBACK,
        start_target_idx=LOOKBACK,
        end_target_idx=len(train_df)
    )

    # 时间序列验证集：取训练序列最后 10% 作为验证集
    val_size = int(len(X_train_all) * VAL_RATIO_IN_TRAIN)
    train_size = len(X_train_all) - val_size

    X_train = X_train_all[:train_size]
    y_train = y_train_all[:train_size]

    X_val = X_train_all[train_size:]
    y_val = y_train_all[train_size:]

    # -----------------------------------------------------
    # 8.8 构造测试序列
    # -----------------------------------------------------
    # 为了让测试集第一个样本也有过去 24 小时历史，
    # 把训练集最后 LOOKBACK 行接到测试集前面。
    # 这不算作弊，因为这是测试时刻之前已经发生的历史数据。

    test_with_history = pd.concat(
        [train_df.iloc[-LOOKBACK:], test_df],
        axis=0
    ).reset_index(drop=True)

    X_test_all_raw = x_scaler.transform(test_with_history[feature_cols])
    y_test_all_raw = y_scaler.transform(test_with_history[[target_col]]).reshape(-1)

    X_test, y_test = build_sequences(
        X_test_all_raw,
        y_test_all_raw,
        lookback=LOOKBACK,
        start_target_idx=LOOKBACK,
        end_target_idx=len(test_with_history)
    )

    print("训练集序列：", X_train.shape, y_train.shape)
    print("验证集序列：", X_val.shape, y_val.shape)
    print("测试集序列：", X_test.shape, y_test.shape)

    # -----------------------------------------------------
    # 8.9 DataLoader
    # -----------------------------------------------------

    train_dataset = AirQualityDataset(X_train, y_train)
    val_dataset = AirQualityDataset(X_val, y_val)
    test_dataset = AirQualityDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------------------------------
    # 8.10 定义模型
    # -----------------------------------------------------

    input_size = X_train.shape[-1]

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    print(model)

    # -----------------------------------------------------
    # 8.11 训练
    # -----------------------------------------------------

    best_val_loss = float("inf")
    best_state = None

    patience = 8
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("\n开始训练")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, "best_lstm_air_quality.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("验证集长期没有提升，提前停止训练。")
            break

    # -----------------------------------------------------
    # 8.12 测试
    # -----------------------------------------------------

    print("\n开始测试")

    model.load_state_dict(best_state)

    test_loss, pred_scaled, true_scaled = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # 反标准化，恢复成真实 PM2.5 数值
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    true = y_scaler.inverse_transform(true_scaled).reshape(-1)

    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)

    print("\n========== 测试集结果 ==========")
    print(f"Test MSE Loss: {test_loss:.6f}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # -----------------------------------------------------
    # 8.13 保存预测结果
    # -----------------------------------------------------

    result_df = pd.DataFrame({
        "true_pollution": true,
        "pred_pollution": pred
    })

    result_df.to_csv("lstm_prediction_result.csv", index=False, encoding="utf-8-sig")
    print("\n预测结果已保存到：lstm_prediction_result.csv")

    # -----------------------------------------------------
    # 8.14 画训练损失曲线
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

    print("损失曲线已保存到：loss_curve.png")

    # -----------------------------------------------------
    # 8.15 画预测曲线
    # -----------------------------------------------------

    show_len = min(300, len(true))

    plt.figure(figsize=(12, 5))
    plt.plot(true[:show_len], label="True PM2.5")
    plt.plot(pred[:show_len], label="Predicted PM2.5")
    plt.xlabel("Time Step")
    plt.ylabel("PM2.5 / Pollution")
    plt.title("LSTM Prediction on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_curve.png", dpi=300)
    plt.show()

    print("预测曲线已保存到：prediction_curve.png")
    print("\n程序运行结束")


# =========================================================
# 9. 真正执行 main 函数
# =========================================================

if __name__ == "__main__":
    main()