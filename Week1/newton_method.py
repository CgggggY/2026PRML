import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pre_process import load_data, mse, predict 

def newton_method(x, y, max_iter=10, tol=1e-12):
    X = np.column_stack([np.ones_like(x), x])
    theta = np.zeros(X.shape[1])
    history = []

    H = (2.0 / len(x)) * (X.T @ X)

    for epoch in range(max_iter):
        pred = X @ theta
        error = pred - y
        grad = (2.0 / len(x)) * (X.T @ error)
        history.append(mse(y, pred))

        step = np.linalg.inv(H) @ grad
        new_theta = theta - step

        if np.max(np.abs(new_theta - theta)) < tol:
            theta = new_theta
            break

        theta = new_theta

    return theta, history, epoch + 1


def main():
    parser = argparse.ArgumentParser(description="Linear regression by Newton's method")
    parser.add_argument("--file", default="Data4Regression.xlsx", help="Path to xlsx file")
    parser.add_argument("--outdir", default="outputs_newton", help="Directory to save figure")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum iterations")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data(args.file)

    theta, history, steps = newton_method(x_train, y_train, max_iter=args.max_iter)
    train_pred = predict(theta, x_train)
    test_pred = predict(theta, x_test)

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    print("=== 牛顿法结果 ===")
    print(f"迭代次数: {steps}")
    print(f"拟合模型: y = {theta[0]:.6f} + {theta[1]:.6f} * x")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")

    x_line = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 400)
    y_line = predict(theta, x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, s=20, label="Train data")
    plt.scatter(x_test, y_test, s=20, label="Test data")
    plt.plot(x_line, y_line, linewidth=2, label="Newton fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression - Newton Method")
    plt.legend()
    plt.tight_layout()
    save_path_fit = os.path.join(args.outdir, "newton_fit.png")
    plt.savefig(save_path_fit, dpi=200)

    plt.figure(figsize=(8, 5))
    plt.plot(history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.title("Newton Method Loss Curve")
    plt.tight_layout()
    save_path_loss = os.path.join(args.outdir, "newton_loss.png")
    plt.savefig(save_path_loss, dpi=200)

    print(f"拟合图像已保存到: {save_path_fit}")
    print(f"损失曲线已保存到: {save_path_loss}")


if __name__ == "__main__":
    main()
