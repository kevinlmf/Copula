import numpy as np
from .data_preprocessing import compute_kendalls_tau

# 导入各维度模型
from .models.low_dim_models import fit_parametric_gaussian_copula
from .models.medium_dim_factor import fit_factor_t_copula
from .models.high_dim_graph_factor import fit_graph_factor_copula
from .models.infinite_dim_dpm import fit_dpm_copula

# 评估函数（可选）
from .evaluation import evaluate_fit


def run_copula_pipeline(Y=None, evaluate=True):
    """
    根据输入数据 Y（n x d），自动选择适当的 Copula 模型并进行估计。
    """
    # ----------------------
    # Step 0: Load default data if not provided
    # ----------------------
    if Y is None:
        Y = np.random.randn(300, 20)  # 默认是中维度

    n, d = Y.shape
    print(f"📐 Data shape: n={n}, d={d}")

    # ----------------------
    # Step 1: Compute dependence structure
    # ----------------------
    tau_matrix = compute_kendalls_tau(Y)
    print("🔍 Kendall’s tau matrix computed.")

    # ----------------------
    # Step 2: Model Selection
    # ----------------------
    if d <= 10:
        print("✅ Using parametric Gaussian Copula (low-dim)")
        model, params, Y_sim = fit_parametric_gaussian_copula(Y)

    elif d <= 50:
        print("✅ Using Factor t-Copula (medium-dim)")
        model, params, Y_sim = fit_factor_t_copula(Y)

    elif d <= 300:
        print("✅ Using Graph-Regularized Factor Copula (high-dim)")
        model, params, Y_sim = fit_graph_factor_copula(Y)

    else:
        print("✅ Using Dirichlet Process Mixture of Factor Copulas (very high-dim)")
        model, params, Y_sim = fit_dpm_copula(Y)

    # ----------------------
    # Step 3: Optional Evaluation
    # ----------------------
    if evaluate:
        evaluate_fit(Y, Y_sim)

    # ----------------------
    # Step 4: Return summary
    # ----------------------
    return model, params, {
        "n": n,
        "d": d,
        "model_type": params.get("type", "Unknown"),
        "tau_shape": tau_matrix.shape
    }
