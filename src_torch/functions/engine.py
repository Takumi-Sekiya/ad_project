import os
import copy
import json
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .callbacks import EarlyStopping

plt.rcParams['font.family'] = ['Arial', 'MS Gothic']
plt.rcParams.update({
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.titlesize': 24
})

def create_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """設定ファイルからPyTorchのOptimizerを生成する"""
    opt_config = config['training']['optimizer']
    opt_name = opt_config['name']
    opt_params = opt_config.get('params', {}).copy()

    # Kerasの 'learning_rate' を PyTorchの 'lr' に読み替える
    if 'learning_rate' in opt_params:
        opt_params['lr'] = opt_params.pop('learning_rate')
        
    # Regulated3DResNetの場合、Optimizer側にL2正則化(Weight Decay)を追加する
    if config['model']['name'] == "Regulated3DResNet" and 'weight_decay' not in opt_params:
        opt_params['weight_decay'] = 1e-4

    try:
        OptimizerClass = getattr(torch.optim, opt_name)
    except AttributeError:
        raise ValueError(f"Optimizer '{opt_name}' not found in torch.optim.")
    
    return OptimizerClass(model.parameters(), **opt_params)

def apply_inverse_scaling(values, target_variable, metadata_path):
    """
    scaling_metadata.json を使用して、スケーリングされた値を元のスケールに復元する。
    """
    if not metadata_path or not os.path.exists(metadata_path):
        print(f" - 情報: メタデータが見つからないため、逆スケーリングはスキップされます。")
        return values

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # target_variable が 'scaled_' で始まる場合、元のカラム名を取得
        base_col = target_variable.replace('scaled_', '')
        
        if base_col not in metadata:
            print(f" - 情報: カラム '{base_col}' のスケーリング情報がメタデータにありません。")
            return values
        
        info = metadata[base_col]
        min_ref = info['min_reference_value']
        max_ref = info['max_reference_value']
        inverted = info.get('inverted', False)

        if min_ref is None or max_ref is None:
            return values

        # 逆変換ロジック
        rescaled_values = np.array(values, dtype=np.float32)
        
        if inverted:
            rescaled_values = 1.0 - rescaled_values
            
        rescaled_values = rescaled_values * (max_ref - min_ref) + min_ref
        return rescaled_values

    except Exception as e:
        print(f"警告: 逆スケーリング中にエラーが発生しました: {e}")
        return values

def save_scatter_plot(y_true, y_pred, title, file_path):
    """散布図生成 (軸ラベルに 'Original Scale' を明記)"""
    r2 = r2_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, c='blue')
    
    # 補助線の範囲をデータの最大値に合わせる
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    ax.plot([min_val, max_val], [min_val, max_val], c='black', linestyle='--')
    
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\n(R2 Score: {r2:.4f})")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
    return fig

def run_training(model: nn.Module, train_ds, test_ds, config: dict):
    """PyTorchのカスタム学習ループ"""
    # 1. デバイス設定 (GPUが利用可能ならCUDA、なければCPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # 2. データローダーの構築
    batch_size = config['training'].get('batch_size', 32)
    # PyTorchのDataLoaderを使用 (num_workers=0でプロセス安全性を確保)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. 損失関数とオプティマイザの準備
    loss_name = config['training'].get('loss', 'mean_squared_error')
    criterion = nn.MSELoss() if loss_name == "mean_squared_error" else nn.MSELoss()
    optimizer = create_optimizer(model, config)

    total_epochs_done = 0

    # 4. 学習ステージ（Warmup, FineTuning等）のループ
    for i, stage_config in enumerate(config['training']['stages']):
        stage_name = stage_config.get('stage_name', f'Stage-{i+1}')
        print(f"\n--- Starting Training: {stage_name} ---")

        # 学習率の更新
        new_lr = stage_config.get('learning_rate')
        if new_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Set learning rate to: {new_lr:.6f}")

        # コールバック・スケジューラの準備
        early_stopper = None
        scheduler = None
        
        for cb in stage_config.get('callbacks', []):
            if cb['name'] == "EarlyStopping":
                params = cb.get('params', {})
                early_stopper = EarlyStopping(
                    patience=params.get('patience', 15),
                    verbose=params.get('verbose', 1),
                    restore_best_weights=params.get('restore_best_weights', True)
                )
            elif cb['name'] == "ReduceLROnPlateau":
                params = cb.get('params', {})
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=params.get('factor', 0.5),
                    patience=params.get('patience', 5),
                    min_lr=params.get('min_lr', 1e-5),
                    #verbose=True  エラーの原因
                )

        epochs_in_stage = stage_config['epochs']
        
        # 5. エポックループ
        for epoch in range(epochs_in_stage):
            current_epoch = total_epochs_done + epoch + 1
            
            # ========== Train Phase ==========
            model.train()
            train_loss_sum = 0.0
            
            for batch_data, targets in train_loader:
                targets = targets.to(device)
                
                # マルチモーダルか単一モーダルかの判定
                if 'numerical_input' in batch_data:
                    outputs = model(batch_data['img_input'].to(device), 
                                    batch_data['numerical_input'].to(device))
                else:
                    outputs = model(batch_data['img_input'].to(device))
                
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item() * targets.size(0)
            
            train_loss_avg = train_loss_sum / len(train_ds)

            # ========== Validation (Test) Phase ==========
            model.eval()
            val_loss_sum = 0.0
            
            with torch.no_grad():
                for batch_data, targets in test_loader:
                    targets = targets.to(device)
                    
                    if 'numerical_input' in batch_data:
                        outputs = model(batch_data['img_input'].to(device), 
                                        batch_data['numerical_input'].to(device))
                    else:
                        outputs = model(batch_data['img_input'].to(device))
                        
                    loss = criterion(outputs, targets)
                    val_loss_sum += loss.item() * targets.size(0)
            
            val_loss_avg = val_loss_sum / len(test_ds)
            
            print(f"Epoch {current_epoch:03d}/{total_epochs_done+epochs_in_stage} - loss: {train_loss_avg:.4f} - val_loss: {val_loss_avg:.4f}")
            
            # MLflowに記録 (autologの代わり)
            mlflow.log_metric("loss", train_loss_avg, step=current_epoch)
            mlflow.log_metric("val_loss", val_loss_avg, step=current_epoch)

            # スケジューラの更新 (ReduceLROnPlateau)
            if scheduler is not None:
                scheduler.step(val_loss_avg)

            # 早期終了のチェック
            if early_stopper is not None:
                early_stopper(val_loss_avg, model)
                if early_stopper.early_stop:
                    print(f"Early stopping triggered at epoch {current_epoch}")
                    break

        # ステージ終了時にベストな重みを復元
        if early_stopper is not None:
            early_stopper.restore(model)
            
        total_epochs_done += epochs_in_stage

    # 6. 学習後モデルの評価・保存
    print("\n--- Final Evaluation & Saving Artifacts ---")
    run_name = config['run_name']
    output_dir = os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 6-1. PyTorch形式(.pth)で重みを保存
    save_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    # 6-2. 予測の実行を関数化
    def get_predictions(loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data, targets in loader:
                if 'numerical_input' in batch_data:
                    outputs = model(batch_data['img_input'].to(device), 
                                    batch_data['numerical_input'].to(device))
                else:
                    outputs = model(batch_data['img_input'].to(device))
                y_pred.append(outputs.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        return np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()

    y_train_true_scaled, y_train_pred_scaled = get_predictions(DataLoader(train_ds, batch_size=config['training'].get('batch_size', 32)))
    y_test_true_scaled, y_test_pred_scaled = get_predictions(DataLoader(test_ds, batch_size=config['training'].get('batch_size', 32)))

    # ★ 逆スケーリングの適用
    target_var = config['data']['target_variable']
    metadata_path = config['data'].get('metadata_path')
    
    y_train_true = apply_inverse_scaling(y_train_true_scaled, target_var, metadata_path)
    y_train_pred = apply_inverse_scaling(y_train_pred_scaled, target_var, metadata_path)
    y_test_true = apply_inverse_scaling(y_test_true_scaled, target_var, metadata_path)
    y_test_pred = apply_inverse_scaling(y_test_pred_scaled, target_var, metadata_path)

    # 6-3. 結果をExcelに保存
    excel_path = os.path.join(output_dir, "predictions_original_scale.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame({'True_Original': y_train_true, 'Predicted_Original': y_train_pred}).to_excel(writer, sheet_name='Train_Data', index=False)
        pd.DataFrame({'True_Original': y_test_true, 'Predicted_Original': y_test_pred}).to_excel(writer, sheet_name='Test_Data', index=False)
    
    # 6-4. プロット図の生成と保存
    train_fig = save_scatter_plot(y_train_true, y_train_pred, "Train Data Scatter Plot", os.path.join(output_dir, "train_scatter_plot.png"))
    mlflow.log_figure(train_fig, "train_scatter_plot.png")

    test_fig = save_scatter_plot(y_test_true, y_test_pred, "Test Data Scatter Plot", os.path.join(output_dir, "test_scatter_plot.png"))
    mlflow.log_figure(test_fig, "test_scatter_plot.png")

    # 6-5. 最終指標の記録
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    r2 = r2_score(y_test_true, y_test_pred)
    mlflow.log_metric("final_test_rmse", rmse)
    mlflow.log_metric("final_test_r2", r2)