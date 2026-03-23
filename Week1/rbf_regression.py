import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pre_process import load_data, mse    

def rbf_features(x, centers, gamma):
    x = x.reshape(-1, 1)
    centers = centers.reshape(1, -1)
    return np.exp(-gamma * (x - centers) ** 2)


def rbf_regression(x, y, num_centers=20, gamma=10.0, reg_lambda=1e-3):
    centers = np.linspace(x.min(), x.max(), num_centers)
    Phi = np.hstack([np.ones((len(x), 1)), rbf_features(x, centers, gamma)])
    I = np.eye(Phi.shape[1])
    theta = np.linalg.solve(Phi.T @ Phi + reg_lambda * I, Phi.T @ y)
    return theta, centers


def predict(theta, centers, gamma, x):
    Phi = np.hstack([np.ones((len(x), 1)), rbf_features(x, centers, gamma)])
    return Phi @ theta


def main():
    parser = argparse.ArgumentParser(description="Nonlinear fitting by RBF regression")
    parser.add_argument("--file", default="Data4Regression.xlsx", help="Path to xlsx file")
    parser.add_argument("--outdir", default="outputs_rbf", help="Directory to save figure")
    parser.add_argument("--num_centers", type=int, default=20, help="Number of RBF centers")
    parser.add_argument("--gamma", type=float, default=10.0, help="RBF gamma")
    parser.add_argument("--reg_lambda", type=float, default=1e-3, help="L2 regularization")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data(args.file)

    theta, centers = rbf_regression(
        x_train, y_train,
        num_centers=args.num_centers,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
    )

    train_pred = predict(theta, centers, args.gamma, x_train)
    test_pred = predict(theta, centers, args.gamma, x_test)

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    print("=== 非线性 RBF 回归结果 ===")
    print(f"RBF centers: {args.num_centers}, gamma: {args.gamma}, lambda: {args.reg_lambda}")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")

    x_line = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 800)
    y_line = predict(theta, centers, args.gamma, x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, s=20, label="Train data")
    plt.scatter(x_test, y_test, s=20, label="Test data")
    plt.plot(x_line, y_line, linewidth=2, label="RBF fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Nonlinear Regression - RBF Basis")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(args.outdir, "rbf_fit.png")
    plt.savefig(save_path, dpi=200)
    print(f"图像已保存到: {save_path}")


if __name__ == "__main__":
    main()
