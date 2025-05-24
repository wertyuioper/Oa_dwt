import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==============================
# 第一部分：阈值函数定义
# ==============================
def hard_threshold(x, T):
    """硬阈值函数"""
    return np.where(np.abs(x) > T, x, 0)


def soft_threshold(x, T):
    """软阈值函数"""
    return np.sign(x) * np.maximum(np.abs(x) - T, 0)


def improved_soft_threshold(x, T, eta=2.5):
    """改进阈值函数（含形状参数eta）"""
    return np.sign(x) * (np.abs(x) - T) / (1 + np.exp(-eta * (np.abs(x) - T)))


# ==============================
# 第二部分：基础阈值函数对比
# ==============================
def plot_threshold_comparison():
    T = 1.0
    x = np.linspace(-3, 3, 500)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x, hard_threshold(x, T), '--', linewidth=2, label='Hard Threshold')
    plt.plot(x, soft_threshold(x, T), '-.', linewidth=2, label='Soft Threshold')
    plt.plot(x, improved_soft_threshold(x, T), '-', linewidth=3, label='Proposed Threshold')

    plt.axvline(T, color='gray', linestyle=':', linewidth=1)
    plt.axvline(-T, color='gray', linestyle=':', linewidth=1)
    plt.text(T + 0.1, 2.2, r'$T=1.0$', rotation=90, fontsize=10)

    plt.title('Threshold Function Comparison')
    plt.xlabel('Wavelet Coefficient Value')
    plt.ylabel('Thresholded Output')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('threshold_compare.pdf', bbox_inches='tight')
    plt.show()


# ==============================
# 第三部分：参数敏感性分析
# ==============================
def parameter_sensitivity_analysis():
    x = np.linspace(-3, 3, 500)
    T_fixed = 1.0
    eta_values = [0.5, 2.5, 5.0]

    plt.figure(figsize=(12, 4))
    for i, eta in enumerate(eta_values):
        plt.subplot(1, 3, i + 1)
        plt.plot(x, hard_threshold(x, T_fixed), '--', label='Hard')
        plt.plot(x, soft_threshold(x, T_fixed), '-.', label='Soft')
        plt.plot(x, improved_soft_threshold(x, T_fixed, eta),
                 '-', label=f'η={eta}')
        plt.title(f"η = {eta}")
        plt.grid(alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig('eta_sensitivity.pdf')
    plt.show()


# ==============================
# 第四部分：去噪性能量化评估
# ==============================
def denoising_performance_eval():
    np.random.seed(42)
    clean_coeff = np.random.laplace(0, 1, 1000)
    noisy_coeff = clean_coeff + 0.5 * np.random.randn(1000)

    def psnr(clean, denoised):
        mse = np.mean((clean - denoised) ** 2)
        return 10 * np.log10(np.max(clean) ** 2 / mse)

    methods = {
        "Hard": (hard_threshold, 1.2),
        "Soft": (soft_threshold, 1.0),
        "Improved": (improved_soft_threshold, 1.0, 2.5)
    }

    results = {}
    for name, params in methods.items():
        if name == "Improved":
            denoised = params[0](noisy_coeff, params[1], params[2])
        else:
            denoised = params[0](noisy_coeff, params[1])
        results[name] = psnr(clean_coeff, denoised)

    plt.figure(figsize=(6, 4))
    plt.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('PSNR (dB)')
    plt.title('Denoising Performance Comparison')
    plt.savefig('psnr_comparison.pdf')
    plt.show()


# ==============================
# 第五部分：3D参数空间分析（优化版）
# ==============================
def parameter_3d_analysis():
    x = np.linspace(-3, 3, 500)
    T_values = np.linspace(0.5, 2.0, 50)
    eta_values = np.linspace(0.5, 5.0, 50)
    X, Y = np.meshgrid(T_values, eta_values)

    # 计算输出均值矩阵
    Z = np.array([[np.mean(improved_soft_threshold(x, T, eta))
                   for T in T_values] for eta in eta_values])

    # 3D可视化
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # 标注传统方法等效点
    ax.scatter(1.2, 5.0, np.mean(hard_threshold(x, 1.2)),
               color='r', s=100, label='Hard Threshold')
    ax.scatter(1.0, 0.5, np.mean(soft_threshold(x, 1.0)),
               color='b', s=100, label='Soft Threshold')

    ax.set_xlabel('Threshold T', fontsize=10)
    ax.set_ylabel('Eta', fontsize=10)
    ax.set_zlabel('Output Mean', fontsize=10)
    ax.legend()
    plt.savefig('3d_analysis_optimized.pdf')
    plt.show()


# ==============================
# 主程序执行
# ==============================
if __name__ == "__main__":
    plot_threshold_comparison()  # 生成阈值函数对比图
    parameter_sensitivity_analysis()  # 参数敏感性分析
    denoising_performance_eval()  # PSNR性能对比
    parameter_3d_analysis()  # 3D参数空间分析