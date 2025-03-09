import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_directory_if_not_exists(directory_path):
    """指定されたディレクトリが存在しない場合、作成する"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def analyze_dataset(file_path, output_path):
    """データセットの基本的な分析を行い、結果をMarkdownファイルに出力する"""
    # データの読み込み
    df = pd.read_csv(file_path)
    
    # 出力用のテキストを準備
    output_text = []
    
    # 基本情報
    output_text.append("# カリフォルニア住宅データの分析レポート\n")
    output_text.append("## 1. データセットの基本情報\n")
    
    # データセットの形状
    output_text.append(f"- データセットの形状: {df.shape[0]}行 × {df.shape[1]}列\n")
    
    # カラム情報
    output_text.append("\n### カラム情報:\n")
    for column in df.columns:
        dtype = str(df[column].dtype)
        n_unique = df[column].nunique()
        output_text.append(f"- {column}: {dtype}, ユニーク値数: {n_unique}\n")
    
    # 基本統計量
    output_text.append("\n## 2. 基本統計量\n```\n")
    output_text.append(df.describe().to_string())
    output_text.append("\n```\n")
    
    # 欠損値の確認
    output_text.append("\n## 3. 欠損値の確認\n")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        output_text.append("欠損値は存在しません。\n")
    else:
        output_text.append("カラムごとの欠損値数:\n```\n")
        output_text.append(null_counts.to_string())
        output_text.append("\n```\n")
    
    # 相関係数
    output_text.append("\n## 4. 数値カラム間の相関係数\n```\n")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    output_text.append(correlation.to_string())
    output_text.append("\n```\n")
    
    # ファイルに書き出し
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_text))

if __name__ == "__main__":
    # プロジェクトのルートディレクトリを取得
    root_dir = Path(__file__).resolve().parents[2]
    
    # 入力ファイルと出力ディレクトリのパスを設定
    input_file = root_dir / "data" / "raw" / "california_housing_data.csv"
    docs_dir = root_dir / "docs"
    output_file = docs_dir / "eda_report.md"
    
    # docsディレクトリが存在しない場合は作成
    create_directory_if_not_exists(docs_dir)
    
    # 分析を実行
    analyze_dataset(str(input_file), str(output_file))
    print(f"分析レポートが {output_file} に保存されました。") 