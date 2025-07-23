import mlflow
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from .callbacks import create_callbacks_from_config

def run_training(model: tf.keras.Model, train_ds, test_ds, config: dict):
    """
    モデルのコンパイル, 学習, 評価, 結果のロギングを行う
    """
    # 1. MLFlowの自動ロギングを有効化
    mlflow.tensorflow.autolog(log_models=True, disable=False)

    # 2. オプティマイザとモデルのコンパイル
    initial_lr = config['training']['stages'][0]['learning_rate']
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
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

    # 5. 学習後の最終評価と手動ロギング
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred = model.predict(test_dataset).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test R2-Score: {r2:.4f}")

    # MLflowにカスタム指標として記録
    mlflow.log_metric("final_test_rmse", rmse)
    mlflow.log_metric("final_test_r2", r2)

    # 6. 結果の可視化とアーティファクト保存
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_pred.max()], 'r--')
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Prediction Scatter Plot (R2: {r2:.3f})")

    # MLflowにプロット画像を保存
    mlflow.log_figure(fig, "prediction_scatter_plot.png")