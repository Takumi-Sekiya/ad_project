import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import time
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.family'] = ['Arial', 'MS Gothic']
plt.rcParams.update({
    'axes.labelsize': 24,     # 軸ラベルのサイズ
    'xtick.labelsize': 18,    # x軸目盛りのサイズ
    'ytick.labelsize': 18,    # y軸目盛りのサイズ
    'axes.titlesize': 24      # タイトルのサイズ
})

input_name = 'hippocampus' #'gray-matter' 'hippocampus'
output_name = '灰白質萎縮度' #'MMSE' '灰白質萎縮度'

with open(f"./data/pickle_data/dataset_{input_name}_{output_name}.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# 画像データを1次元ベクトルに変換
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("-" * 40)


def run_optuna_evaluation(model_name, n_trials, X_train, y_train, X_test, y_test):
    print(f"===== Evaluating {model_name} with Optuna =====")

    n_samples_in_cv_train = int(len(X_train) * 0.8) 
    max_n_components = min(n_samples_in_cv_train, X_train.shape[1])
    print(f"Max n_components for PCA is set to: {max_n_components}")

    # --- 1. 目的関数の定義 ---
    def objective(trial):
        # --- 2. ハイパーパラメータの探索空間を定義 ---
        # PCAの主成分数
        n_components = trial.suggest_int('n_components', 5, max_n_components)

        if model_name == 'Lasso':
            alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
            model = Lasso(alpha=alpha, max_iter=20000)
        
        elif model_name == 'SVR':
            C = trial.suggest_float('C', 1e-1, 1e2, log=True)
            gamma = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
            model = SVR(kernel='rbf', C=C, gamma=gamma)
            
        elif model_name == 'LGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 70),
                'random_state': 42,
                'n_jobs': 1, # Optunaの並列化と競合しないよう1に設定
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
        
        else:
            raise ValueError("Unsupported model_name")

        # パイプラインを構築
        pipeline = Pipeline([
            ('pca', PCA(n_components=n_components)),
            ('regressor', model)
        ])

        # クロスバリデーションでスコアを評価 (n_jobs=-1で並列化)
        # Optunaは最小化が基本なので、MSEをそのまま返す
        score = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        ).mean()

        return score

    # --- 3. Optunaによる学習の実行 ---
    # スコア (負のMSE) を最大化する方向に学習
    study = optuna.create_study(direction='maximize')
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)
    end_time = time.time()
    
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    print(f"  Value (Negative MSE): {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # --- 4. 最適化されたモデルで最終評価 ---
    print("\n--- Training final model with best params and evaluating on test set ---")
    
    # 最適なパラメータを取得
    best_params = study.best_params
    best_n_components = best_params.pop('n_components')
    
    # 最適なモデルを再構築
    if model_name == 'Lasso':
        final_model = Lasso(**best_params, max_iter=10000)
    elif model_name == 'SVR':
        final_model = SVR(kernel='rbf', **best_params)
    elif model_name == 'LGBM':
        final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
        
    final_pipeline = Pipeline([
        ('pca', PCA(n_components=best_n_components)),
        ('regressor', final_model)
    ])
    
    # 訓練データ全体で学習
    final_pipeline.fit(X_train, y_train)
    
    # テストデータで予測と評価
    y_pred = final_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test Set Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Set R-squared (R2): {r2:.4f}")
    print("=" * (40 + len(model_name)))
    print("\n")

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, c="blue")
    plt.plot([0,max(y_test)*1.05], [0,max(y_test)*1.05], c="black")
    plt.tick_params(direction="in")
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(f"{input_name} → {output_name}")
    plt.savefig(f"./output/graph_{model_name}/{input_name}_{output_name}_test.png")

    y_pred = final_pipeline.predict(X_train)

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_train, y_pred, c="blue")
    plt.plot([0,max(y_train)*1.05], [0,max(y_train)*1.05], c="black")
    plt.tick_params(direction="in")
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(f"{input_name} → {output_name}")
    plt.savefig(f"./output/graph_{model_name}/{input_name}_{output_name}_train.png")



# --- 各モデルの評価を実行 ---
# 試行回数 (n_trials) はマシンスペックや許容時間に応じて調整してください
N_TRIALS = 50 

run_optuna_evaluation('LGBM', n_trials=N_TRIALS, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
run_optuna_evaluation('Lasso', n_trials=N_TRIALS, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
run_optuna_evaluation('SVR', n_trials=N_TRIALS, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)