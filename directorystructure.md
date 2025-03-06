root_directory/
├── config/
│   ├── model_config.yaml      # モデルの基本設定
│   └── experiment_configs/    # 実験ごとの設定
│       ├── exp001.yaml       # 実験001の設定
│       └── exp002.yaml       # 実験002の設定
│
├── data/
│   ├── raw/                  # 生データ
│   │   └── california_housing_data.csv
│   ├── interim/             # 中間データ
│   │   └── preprocessed/    # 前処理済みデータ
│   └── processed/           # 最終的な学習データ
│
├── src/
│   ├── features/
│   │   ├── feature_engineering.py # 特徴量エンジニアリング
│   │   └── preprocessing.py       # 前処理用の関数
│   │
│   ├── models/
│   │   ├── train_model.py         # モデルの訓練
│   │   └── predict.py             # 予測用の関数
│   │
│   ├── utils/
│   │   ├── data_utils.py           # データのユーティリティ関数
│   │   └── evaluation.py           # 評価指標の計算
│   │ 
├── experiments/           # 実験結果の保存
│   ├── exp001/           # 実験001のディレクトリ
│   │   ├── model/       # モデルの保存
│   │   ├── predictions/ # 予測結果
│   │   └── metrics.json # 評価指標
│   └── exp002/
│
├── docs/                 # ドキュメント
│   ├── eda_report.md    # EDA結果
│   └── results.md      # 結果のまとめ
|
├── logs/                 # ログ
│   ├── logs_exp001.txt         # ログの保存
│   └── logs_exp002.txt
|
├── .gitignore          # Git除外設定
├── README.md           # プロジェクト概要
└── requirements.txt    # 依存パッケージ