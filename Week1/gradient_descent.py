import argparse
import os
import numpy as np  
import matplotlib.pyplot as plt
from pre_process import load_data, mse, predict 

def BGD(x, y, lr=0.01, max_iter=5000, tol=1e-10):
    w0, w1 = 0.0, 0.0
    history = []
    n = len(x)

    for epoch in range(max_iter):
        y_pred = w0 + w1 * x
        error = y_pred - y  

        grad_w0 = 2.0 * np.mean(error)
        grad_w1 = 2.0 * np.mean(error * x)

        new_w0 = w0 - lr * grad_w0
        new_w1 = w1 - lr * grad_w1

        history.append(mse(y, y_pred))

        if max(abs(new_w0 - w0), abs(new_w1 - w1)) < tol:
            w0, w1 = new_w0, new_w1
            break

        w0, w1 = new_w0, new_w1

    return np.array([w0, w1]), history, epoch + 1


def main():
    parser = argparse.ArgumentParser(description="Linear regression by gradient descent")
    parser.add_argument("--file", default="Data4Regression.xlsx", help="Path to xlsx file")
    parser.add_argument("--outdir", default="outputs_gd", help="Directory to save figure")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--max_iter", type=int, default=5000, help="Maximum iterations")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data(args.file)

    theta, history, steps = BGD(x_train, y_train, lr=args.lr, max_iter=args.max_iter)
    train_pred = predict(theta, x_train)
    test_pred = predict(theta, x_test)

    train_mse = mse(y_train, train_pred)    
    test_mse = mse(y_test, test_pred)

    print("=== 梯度下降法结果 ===")
    print(f"迭代次数: {steps}")
    print(f"拟合模型: y = {theta[0]:.6f} + {theta[1]:.6f} * x")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")

    x_line = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 400)
    y_line = predict(theta, x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, s=20, label="Train data")
    plt.scatter(x_test, y_test, s=20, label="Test data")
    plt.plot(x_line, y_line, linewidth=2, label="GD fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression - Gradient Descent")
    plt.legend()
    plt.tight_layout()
    save_path_fit = os.path.join(args.outdir, "gradient_descent_fit.png")
    plt.savefig(save_path_fit, dpi=200)

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.title("Gradient Descent Loss Curve")
    plt.tight_layout()
    save_path_loss = os.path.join(args.outdir, "gradient_descent_loss.png")
    plt.savefig(save_path_loss, dpi=200)

    print(f"拟合图像已保存到: {save_path_fit}")
    print(f"损失曲线已保存到: {save_path_loss}")


if __name__ == "__main__":
    main()
