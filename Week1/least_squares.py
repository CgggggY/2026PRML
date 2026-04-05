import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pre_process import load_data, mse, predict 

def least_squares(x, y):
    X = np.column_stack([np.ones_like(x), x])
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta


def main():
    parser = argparse.ArgumentParser(description="Linear regression by least squares")
    parser.add_argument("--file", default="Data4Regression.xlsx", help="Path to xlsx file")
    parser.add_argument("--outdir", default="outputs_ls", help="Directory to save figure")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data(args.file)

    theta = least_squares(x_train, y_train)
    train_pred = predict(theta, x_train)
    test_pred = predict(theta, x_test)

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    print("=== 最小二乘法结果 ===")
    print(f"拟合模型: y = {theta[0]:.6f} + {theta[1]:.6f} * x")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")

    x_line = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 400)
    y_line = predict(theta, x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, s=20, label="Train data")
    plt.scatter(x_test, y_test, s=20, label="Test data")
    plt.plot(x_line, y_line, linewidth=2, label="Least squares fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression - Least Squares")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(args.outdir, "least_squares_fit.png")
    plt.savefig(save_path, dpi=200)
    print(f"图像已保存到: {save_path}")


if __name__ == "__main__":
    main()
