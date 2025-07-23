import argparse
import yaml
import mlflow
import tensorflow as tf

from src.utils import set_seed
from src.data_loader import get_datasets
from src.models import build_model
from src.engine import run_training

def main(config_path: str):
    # 1. 設定ファイルを読み込む
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. MLflowの実験を開始
    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run(run_name=config['run_name']):
        # 設定ファイルの内容をパラメータとして記録
        mlflow.log_params(config['data'])
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])

        # 3. 再現性のための乱数固定
        set_seed(config['environment']['seed'])

        # 4. データセットを取得
        train_ds, test_ds = get_datasets(config)

        # 5. モデルを構築
        img_shape = next(iter(train_ds))[0].shape
        model = build_model(input_shape=img_shape, config=config)
        model.summary()

        # 6. 学習と評価を実行
        run_training(model, train_ds, test_ds, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    main(args.config)

# python script/train.py --config script/config/exp001_gray-matter_mmse.yaml