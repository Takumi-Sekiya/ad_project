import os
import mlflow
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .callbacks import create_callbacks_from_config

plt.rcParams['font.family'] = ['Arial', 'MS Gothic']
plt.rcParams.update({
    'axes.labelsize': 24,     # 軸ラベルのサイズ
    'xtick.labelsize': 18,    # x軸目盛りのサイズ
    'ytick.labelsize': 18,    # y軸目盛りのサイズ
    'axes.titlesize': 24      # タイトルのサイズ
})

def create_optimizer(config: dict) -> tf.keras.optimizers.Optimizer:
    """
    設定ファイルからOptimizerを動的に生成する
    """
    opt_config = config['training']['optimizer']
    opt_name = opt_config['name']
    opt_params = opt_config.get('params', {})

    try:
        OptimizerClass = getattr(tf.keras.optimizers, opt_name)
    except AttributeError:
        raise ValueError(f"Optimizer '{opt_name}' not found in tf.keras.optimizers.")
    
    return OptimizerClass(**opt_params)

def save_scatter_plot(y_true, y_pred, title, file_path):
    """
    散布図を生成し, 指定されたパスに保存する
    """
    r2 = r2_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=1, c='blue')
    ax.plot([0, y_true.max()*1.05], [0, y_true.max()*1.05], c='black')
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{title}\n(R2 Score: {r2:.4f})")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
    return fig


def run_training(model: tf.keras.Model, train_ds, test_ds, config: dict):
    """
    モデルのコンパイル, 学習, 評価, 結果のロギングを行う
    """
    # 1. MLFlowの自動ロギングを有効化
    mlflow.tensorflow.autolog(log_models=True, disable=False)

    # 2. オプティマイザとモデルのコンパイル
    optimizer = create_optimizer(config)
    model.compile(optimizer=optimizer, loss=config['training']['loss'])

    # 3. データセットのバッチ化とシャッフル
    batch_size = config['training'].get('batch_size', 32)
    train_dataset = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 4. モデルの学習
    total_epochs_done = 0
    for i, stage_config in enumerate(config['training']['stages']):
        stage_name = stage_config.get('stage_name', f'Stage-{i+1}')
        print(f"\n--- Starting Training: {stage_name} ---")

        # 4-1. 学習率の動的更新
        new_lr = stage_config['learning_rate']
        model.optimizer.learning_rate.assign(new_lr)
        print(f"Set learning rate to: {model.optimizer.learning_rate.numpy():.6f}")

        # 4-2. コールバックを動的に生成
        stage_callbacks = create_callbacks_from_config(stage_config.get('callbacks', []))

        # 4-3. model.fitの実行
        epochs_in_stage = stage_config['epochs']
        model.fit(
            train_dataset,
            epochs=total_epochs_done + epochs_in_stage,
            initial_epoch=total_epochs_done,
            validation_data = test_dataset,
            callbacks=stage_callbacks,
            verbose=2
        )
        total_epochs_done += epochs_in_stage

    # 5. 学習後モデルの評価
    print("\n--- Final Evaluation & Saving Artifacts ---")

    # 5-1. ローカルの出力ディレクトリを作成
    run_name = config['run_name']
    output_dir = os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {output_dir}")

    # 5-2. 訓練済みモデルの保存
    model.save(os.path.join(output_dir, "model"))

    # 5-3. 訓練データとテストデータで予測を実行
    y_train_true = np.concatenate([y for _, y in train_ds.batch(batch_size)])
    y_train_pred = model.predict(train_ds.batch(batch_size)).flatten()

    y_test_true = np.concatenate([y for _, y in test_dataset])
    y_test_pred = model.predict(test_dataset).flatten()

    # 5-4. Excelファイルに予測結果を保存
    excel_path = os.path.join(output_dir, "predictions.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred})
        train_df.to_excel(writer, sheet_name='Train_Data', index=False)

        test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred})
        test_df.to_excel(writer, sheet_name='Test_Data', index=False)
    
    # 5-5. プロット図を生成し, ローカルとMLflowに保存
    train_plot_path = os.path.join(output_dir, "train_scatter_plot.png")
    train_fig = save_scatter_plot(y_train_true, y_train_pred, "Train Data Scatter Plot", train_plot_path)
    mlflow.log_figure(train_fig, "train_scatter_plot.png")

    test_plot_path = os.path.join(output_dir, "test_scatter_plot.png")
    test_fig = save_scatter_plot(y_test_true, y_test_pred, "Test Data Scatter Plot", test_plot_path)
    mlflow.log_figure(test_fig, "test_scatter_plot.png")

    # 5-6. MLflowに最終指標を記録
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    r2 = r2_score(y_test_true, y_test_pred)
    mlflow.log_metric("final_test_rmse", rmse)
    mlflow.log_metric("final_test_r2", r2)