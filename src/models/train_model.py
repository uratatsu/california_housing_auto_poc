import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import yaml
import json
from pathlib import Path
import sys

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parents[2]))
from src.features.feature_engineering import prepare_features

def load_config():
    """設定ファイルを読み込む"""
    config_path = Path(__file__).parents[2] / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """データを読み込む"""
    data_path = Path(__file__).parents[2] / "data" / "raw" / "california_housing_data.csv"
    return pd.read_csv(data_path)

def train_and_evaluate():
    """モデルの訓練と評価を行う"""
    # 設定の読み込み
    config = load_config()
    
    # データの読み込み
    df = load_data()
    
    # ターゲットの分離
    X = df.drop('HousingPrices', axis=1)
    y = df['HousingPrices']
    
    # 訓練データとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['train_test_split_ratio'],
        random_state=config['data']['random_state']
    )
    
    # 特徴量エンジニアリング
    X_train_featured, scaler = prepare_features(X_train, is_training=True)
    X_test_featured, _ = prepare_features(X_test, scaler=scaler, is_training=False)
    
    # クロスバリデーションの設定
    kf = KFold(
        n_splits=config['training']['cv_folds'],
        shuffle=True,
        random_state=config['data']['random_state']
    )
    
    # モデルのパラメータ
    model_params = config['model']['params'].copy()
    num_boost_round = model_params.pop('n_estimators')
    
    # クロスバリデーションでの訓練と評価
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_featured), 1):
        # データの分割
        X_fold_train = X_train_featured.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_featured.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # データセットの作成
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val)
        
        # モデルの訓練
        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data]
        )
        
        # 予測と評価
        val_pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        cv_scores.append(rmse)
        models.append(model)
        
        print(f"Fold {fold} - RMSE: {rmse:.4f}")
    
    # テストデータでの評価
    test_predictions = np.zeros(len(X_test))
    for model in models:
        test_predictions += model.predict(X_test_featured) / len(models)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    # 結果の保存
    results = {
        'cv_scores': [float(score) for score in cv_scores],
        'cv_mean_rmse': float(np.mean(cv_scores)),
        'cv_std_rmse': float(np.std(cv_scores)),
        'test_rmse': float(test_rmse)
    }
    
    # 実験ディレクトリの作成
    experiment_dir = Path(__file__).parents[2] / "experiments" / "exp001"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 結果の保存
    with open(experiment_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # モデルの保存
    for i, model in enumerate(models, 1):
        model.save_model(str(experiment_dir / f"model_fold_{i}.txt"))
    
    print("\n=== 実験結果 ===")
    print(f"CV平均RMSE: {results['cv_mean_rmse']:.4f} ± {results['cv_std_rmse']:.4f}")
    print(f"テストRMSE: {results['test_rmse']:.4f}")
    
    return results

if __name__ == "__main__":
    train_and_evaluate() 