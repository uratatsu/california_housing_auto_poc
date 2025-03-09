import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path

def load_config():
    """設定ファイルを読み込む"""
    config_path = Path(__file__).parents[2] / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリングを行う"""
    df_new = df.copy()
    
    # 部屋あたりの寝室の割合
    df_new['bedroom_ratio'] = df_new['AveBedrms'] / df_new['AveRooms']
    
    # 人口密度（世帯人数あたりの人口）
    df_new['population_density'] = df_new['Population'] / df_new['AveOccup']
    
    # 緯度経度から地域特性を作成
    df_new['location_cluster'] = (df_new['Latitude'] * df_new['Longitude']).round(1)
    
    # 収入と部屋数の相互作用
    df_new['income_rooms'] = df_new['MedInc'] * df_new['AveRooms']
    
    return df_new

def scale_features(df: pd.DataFrame, scaler=None, is_training=True) -> tuple:
    """特徴量のスケーリングを行う"""
    config = load_config()
    features_to_scale = config['feature_engineering']['features_to_use']
    
    if is_training:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features_to_scale])
    else:
        scaled_features = scaler.transform(df[features_to_scale])
    
    scaled_df = pd.DataFrame(
        scaled_features,
        columns=features_to_scale,
        index=df.index
    )
    
    # スケーリングしない特徴量があれば結合
    non_scaled_features = [col for col in df.columns if col not in features_to_scale]
    if non_scaled_features:
        scaled_df = pd.concat([scaled_df, df[non_scaled_features]], axis=1)
    
    return scaled_df, scaler

def prepare_features(df: pd.DataFrame, scaler=None, is_training=True) -> tuple:
    """特徴量エンジニアリングのメイン関数"""
    config = load_config()
    
    # 特徴量作成
    df_featured = create_features(df)
    
    # スケーリング
    if config['feature_engineering']['scaling']:
        df_featured, scaler = scale_features(df_featured, scaler, is_training)
    
    return df_featured, scaler 