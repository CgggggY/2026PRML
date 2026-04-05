import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def load_data(file_path: str):
    xls = pd.ExcelFile(file_path)
    train_df = pd.read_excel(xls, sheet_name=0)
    test_df = pd.read_excel(xls, sheet_name=1)

    x_train = train_df.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
    y_train = train_df.iloc[:, 1].to_numpy(dtype=float)
    x_test = test_df.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
    y_test = test_df.iloc[:, 1].to_numpy(dtype=float)

    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser(description="MLP nonlinear regression for Data4Regression")
    parser.add_argument("--file", type=str, default="Data4Regression.xlsx", help="Excel file path")
    args = parser.parse_args()

    # 1. 读取数据
    x_train, y_train, x_test, y_test = load_data(args.file)

    # 2. 标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_std = x_scaler.fit_transform(x_train)
    x_test_std = x_scaler.transform(x_test)
    y_train_std = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # 3. 定义并训练 MLP 模型
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-2,
        max_iter=5000,
        random_state=42,
        early_stopping=False
    )
    model.fit(x_train_std, y_train_std)

    # 4. 预测
    y_train_pred_std = model.predict(x_train_std)
    y_test_pred_std = model.predict(x_test_std)

    y_train_pred = y_scaler.inverse_transform(y_train_pred_std.reshape(-1, 1)).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_std.reshape(-1, 1)).ravel()

    # 5. 计算误差
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # 6. 输出结果
    print("===== MLP Nonlinear Regression =====")
    print(f"Hidden layers: {model.hidden_layer_sizes}")
    print(f"Activation   : {model.activation}")
    print(f"Train MSE    : {train_mse:.6f}")
    print(f"Test MSE     : {test_mse:.6f}")

    # 7. 可视化
    x_all = np.vstack([x_train, x_test])
    x_min, x_max = x_all.min(), x_all.max()
    x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    x_plot_std = x_scaler.transform(x_plot)
    y_plot_std = model.predict(x_plot_std)
    y_plot = y_scaler.inverse_transform(y_plot_std.reshape(-1, 1)).ravel()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, label="Train Data", s=28)
    plt.scatter(x_test, y_test, label="Test Data", s=28)
    plt.plot(x_plot, y_plot, linewidth=2, label="MLP Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MLP Nonlinear Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()