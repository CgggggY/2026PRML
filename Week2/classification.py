import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# 输出目录（与脚本同目录）
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_TXT = OUTPUT_DIR / "classification_results.txt"
METRICS_FIG = OUTPUT_DIR / "model_metrics_comparison.png"


class _TeeStdout:
    """同时写入控制台与文件。"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# =========================
# 1. 生成 3D make-moons 数据
# =========================
def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([
        np.column_stack([x, y, z]),
        np.column_stack([-x, y - 1, -z])
    ])
    labels = np.hstack([
        np.zeros(n_samples),
        np.ones(n_samples)
    ])

    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels


# =========================
# 2. 可视化数据
# =========================
def plot_3d_data(X, y, title="3D Make Moons"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o', s=20)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# 3. 评估函数
# =========================
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-score": f1
    }


# =========================
# 4. 主程序
# =========================
def _main_inner():
    # 训练集：总共1000个点，每类500个
    X_train, y_train = make_moons_3d(n_samples=500, noise=0.2, random_state=42)

    # 测试集：总共500个点，每类250个
    X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=123)

    print("Train set size:", X_train.shape[0])
    print("Test set size :", X_test.shape[0])

    # 可视化训练集
    plot_3d_data(X_train, y_train, title="Training Data (3D Make Moons)")
    plot_3d_data(X_test, y_test, title="Testing Data (3D Make Moons)")

    results = []

    # 1) Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    results.append(evaluate_model("Decision Tree", dt_model, X_train, y_train, X_test, y_test))

    # 2) AdaBoost + Decision Tree
    base_tree = DecisionTreeClassifier(
        max_depth=2,
        random_state=42
    )
    ada_model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )
    results.append(evaluate_model("AdaBoost + Decision Tree", ada_model, X_train, y_train, X_test, y_test))

    # 3) SVM - Linear Kernel
    svm_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=1.0))
    ])
    results.append(evaluate_model("SVM (Linear Kernel)", svm_linear, X_train, y_train, X_test, y_test))

    # 4) SVM - Polynomial Kernel
    svm_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="poly", degree=3, C=1.0, gamma="scale"))
    ])
    results.append(evaluate_model("SVM (Polynomial Kernel)", svm_poly, X_train, y_train, X_test, y_test))

    # 5) SVM - RBF Kernel
    svm_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])
    results.append(evaluate_model("SVM (RBF Kernel)", svm_rbf, X_train, y_train, X_test, y_test))

    # 汇总打印
    print("\n================ Summary ================")
    print(f"{'Model':30s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
    for r in results:
        print(f"{r['Model']:30s} {r['Accuracy']:10.4f} {r['Precision']:10.4f} {r['Recall']:10.4f} {r['F1-score']:10.4f}")

    plot_metrics_comparison(results)


def plot_metrics_comparison(results, save_path=None):
    """
    四个指标各一幅子图：横轴为模型，纵轴为分数。
    """
    if save_path is None:
        save_path = METRICS_FIG

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    models = [r["Model"] for r in results]
    n = len(models)
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_titles = ["准确率 (Accuracy)", "精确率 (Precision)", "召回率 (Recall)", "F1 分数 (F1-score)"]

    # 每模型一条高饱和渐变色（turbo）
    try:
        turbo = plt.colormaps["turbo"]
    except (AttributeError, KeyError):
        turbo = plt.cm.get_cmap("turbo")
    bar_colors = [turbo(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0c0f14")
    axes = axes.ravel()

    for ax, mkey, mtitle in zip(axes, metric_keys, metric_titles):
        vals = [float(r[mkey]) for r in results]
        x = np.arange(n)
        bars = ax.bar(
            x,
            vals,
            color=bar_colors,
            edgecolor="#e8eef8",
            linewidth=1.0,
            width=0.62,
            alpha=0.92,
        )
        ax.set_facecolor("#141a24")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=22, ha="right", color="#c5d0e0", fontsize=9)
        ax.set_ylabel("分数", color="#e2e8f0", fontsize=11)
        ax.set_title(mtitle, color="#f1f5f9", fontsize=13, fontweight="bold", pad=14)
        ax.tick_params(axis="y", colors="#94a3b8", labelsize=10)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, linestyle=(0, (4, 6)), color="#334155", alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#3d4f66")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(v + 0.025, 1.02),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=9,
                fontweight="semibold",
            )

    fig.suptitle(
        "3D Make Moons 分类 — 各模型指标对比",
        fontsize=16,
        color="#f8fafc",
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, wspace=0.22, hspace=0.38)
    fig.savefig(
        save_path,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(f"\n指标对比图已保存: {save_path}")


def main():
    RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_TXT, "w", encoding="utf-8") as logf:
        old_stdout = sys.stdout
        sys.stdout = _TeeStdout(old_stdout, logf)
        try:
            _main_inner()
        finally:
            sys.stdout = old_stdout
    print(f"\n运行结果已写入: {RESULTS_TXT}")


if __name__ == "__main__":
    main()
