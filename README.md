# カリフォルニア住宅価格予測 PoC

このリポジトリは、Cursor AIアシスタントによって実装された機械学習PoCプロジェクトです。Cursorのプロジェクトルール（`global.mdc`）に従って、段階的に実装されています。

## プロジェクト概要

カリフォルニア住宅データセットを使用して、住宅価格の予測モデルを構築するPoCです。LightGBMを用いた回帰モデルを実装し、RMSEによる評価を行っています。

### 実装ステータス

- [x] データの確認とターゲットの確認
- [x] 実装計画の立案
- [x] 基本モデルの実装と評価
- [x] 結果のドキュメント化

## ディレクトリ構成

```
root_directory/
├── config/
│   └── model_config.yaml      # モデルの設定ファイル
│
├── data/
│   ├── raw/                  # 生データ
│   │   └── california_housing_data.csv
│   ├── interim/             # 中間データ
│   └── processed/           # 最終的な学習データ
│
├── src/
│   ├── features/
│   │   └── feature_engineering.py # 特徴量エンジニアリング
│   │
│   ├── models/
│   │   └── train_model.py         # モデルの訓練
│   │
│   └── utils/               # ユーティリティ関数
│
├── experiments/            # 実験結果の保存
│   └── exp001/            # 実験001の結果
│
├── docs/                  # ドキュメント
│   ├── eda_report.md     # EDA結果
│   └── experiment_report.md  # 実験レポート
│
├── requirements.txt       # 依存パッケージ
└── README.md             # 本ファイル
```

## 技術スタック

- Python 3.12.4
- pandas
- numpy
- scikit-learn
- lightgbm
- pyyaml

## 主な機能

1. **特徴量エンジニアリング**
   - 基本特徴量の前処理
   - 派生特徴量の作成
   - スケーリング処理

2. **モデリング**
   - LightGBMによる回帰モデル
   - 5分割クロスバリデーション
   - アンサンブル予測

3. **評価**
   - RMSE（平均二乗誤差の平方根）による評価
   - クロスバリデーションスコア
   - テストデータでの性能評価

## 実験結果

初期実験（exp001）では以下の結果を達成:
- CV平均RMSE: 0.4877 ± 0.0121
- テストRMSE: 0.4834

詳細な分析結果は `docs/experiment_report.md` を参照してください。

## セットアップと実行方法

1. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

2. モデルの訓練:
```bash
python src/models/train_model.py
```

## 今後の展開

詳細な改善案と今後の方向性については、`docs/experiment_report.md`の「改善案と今後の方向性」セクションを参照してください。

## 実装について

このプロジェクトは、Cursor AIアシスタントによって、`global.mdc`で定義された機械学習PoCのルールに従って実装されています。実装プロセスは以下の手順で行われました：

1. データの確認とEDA
2. 実装計画の立案
3. 特徴量エンジニアリングの実装
4. モデルの構築と評価
5. 実験結果のドキュメント化

各ステップは、プロジェクトルールに基づいて慎重に実行され、適切なドキュメント化が行われています。 