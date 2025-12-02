#!/usr/bin/env python3

import glob
import pandas as pd
from deepface import DeepFace
import os
import warnings
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# DeepFaceのログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# 教師データが保存されている親ディレクトリ
DATA_DIR = "dataset/"

# 分析するプロパティ
ACTIONS = ('age', 'gender', 'emotion', 'race')
all_features = []

# 分析するプロパティ
ACTIONS = ('age', 'gender', 'emotion', 'race')
all_features = []

# 失敗した画像を保存するディレクトリを作成
FALSES_DIR = "falses"
os.makedirs(FALSES_DIR, exist_ok=True)

# 既存のCSVファイルから処理済み画像のパスを取得
processed_images = set()
if os.path.exists("facial_features_dataset.csv"):
    try:
        existing_df = pd.read_csv("facial_features_dataset.csv")
        if 'image_path' in existing_df.columns:
            processed_images = set(existing_df['image_path'].tolist())
            print(f"Found {len(processed_images)} already processed images.")
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")

# DATA_DIR内の各フォルダ (group_A, group_B, ...) をループ
for label_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"--- Processing folder: {label_name} ---")

    # フォルダ内の全画像ファイルを取得（大文字小文字を区別しない）
    import glob
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    for img_path in image_paths:
        # 既に処理済みの画像はスキップ
        if img_path in processed_images:
            print(f"Skipping already processed: {img_path}")
            continue

        try:
            # DeepFaceで顔分析を実行
            analysis_list = DeepFace.analyze(
                img_path=img_path,
                actions=ACTIONS,
                enforce_detection=True # 顔が検出されない場合はエラーにする
            )

            # 必要な特徴量と、正解ラベル（フォルダ名）を辞書として保存
            features = {
                'image_path': img_path,  # 画像パスを追加
                'age': face_data['age'],
                'gender': face_data['dominant_gender'],
                'emotion': face_data['dominant_emotion'],
                'race': face_data['dominant_race'],
                'label': label_name  # フォルダ名を正解ラベルとして使用
            }

            # 1枚の画像から1つの顔だけを使う (analyzeはリストを返す)
            if analysis_list and isinstance(analysis_list, list):
                face_data = analysis_list[0]

                # 必要な特徴量と、正解ラベル（フォルダ名）を辞書として保存
                features = {
                    'age': face_data['age'],
                    'gender': face_data['dominant_gender'],
                    'emotion': face_data['dominant_emotion'],
                    'race': face_data['dominant_race'],
                    'label': label_name  # フォルダ名を正解ラベルとして使用
                }
                all_features.append(features)

        except Exception as e:
            # 顔が検出できなかった場合など
            print(f"Skipping {img_path}: {e}")
            # 失敗した画像をfalsesディレクトリに移動
            try:
                # ファイル名を取得
                filename = os.path.basename(img_path)
                # falsesディレクトリへのパス
                dest_path = os.path.join(FALSES_DIR, filename)
                # ファイル名が重複する場合は、元のディレクトリ名を追加
                if os.path.exists(dest_path):
                    label_name = os.path.basename(os.path.dirname(img_path))
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(FALSES_DIR, f"{label_name}_{name}{ext}")
                # ファイルをコピー
                shutil.copy2(img_path, dest_path)
                # 元のファイルを削除
                os.remove(img_path)
                print(f"Moved failed image to: {dest_path}")
            except Exception as move_error:
                print(f"Failed to move image {img_path}: {move_error}")

# 抽出した全特徴量をPandas DataFrameに変換
df = pd.DataFrame(all_features)

# 既存のCSVファイルがある場合は結合
if os.path.exists("facial_features_dataset.csv") and not df.empty:
    try:
        existing_df = pd.read_csv("facial_features_dataset.csv")
        # 既存データと新規データを結合（重複を除去）
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=['image_path'], keep='last')  # 同じ画像パスは最新のものを保持
        print(f"Combined with existing data. Total: {len(df)} faces.")
    except Exception as e:
        print(f"Warning: Could not combine with existing data: {e}")
        df = df
else:
    df = df

# 結果の確認と保存
print("\n--- Extracted Features DataFrame ---")
print(df.head())
print(f"\nTotal {len(df)} faces processed.")

# (任意) 抽出した特徴量をCSVファイルとして保存しておくと、次回から高速に処理できる
df.to_csv("facial_features_dataset.csv", index=False)

# (CSVから読み込む場合)
# df = pd.read_csv("facial_features_dataset.csv")

if df.empty:
    print("No data to train. Exiting.")
else:
    # 1. 特徴量(X)と正解ラベル(y)に分離
    X = df.drop('label', axis=1)
    y = df['label']

    # データの分布を確認
    print("\n--- Class Distribution ---")
    print(y.value_counts())
    print(f"\nMinimum samples per class: {y.value_counts().min()}")

    # 2. 学習データとテストデータに分割 (80%で学習, 20%で評価)
    # 各クラスに最低2つのサンプルがある場合のみstratifyを使用
    if y.value_counts().min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("\nUsing stratified split.")
    else:
        print(f"\nWarning: Not enough samples per class for stratified split (minimum: {y.value_counts().min()}). Using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 前処理パイプラインの定義
    # 数値カラム
    numeric_features = ['age']
    numeric_transformer = StandardScaler()

    # カテゴリカラム
    categorical_features = ['gender', 'emotion', 'race']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') # 学習時に見なかったカテゴリは無視

    # ColumnTransformerで、カラムごとに異なる前処理を定義
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. モデルの定義 (ランダムフォレスト分類器を使用)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # 5. [重要] 前処理と分類器をパイプラインとして連結
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # ステップ1: 前処理
        ('classifier', classifier)      # ステップ2: 分類
    ])

    # 6. モデルの学習
    print("\n--- Training the model ---")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 7. モデルの評価 (テストデータを使用)
    print("\n--- Evaluating the model ---")
    y_pred = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # zero_division=0 は、テストデータにそのクラスが1つも無かった場合の警告を抑制します
    print(classification_report(y_test, y_pred, zero_division=0))

    # (任意) 学習済みモデルをファイルに保存
    import joblib
    joblib.dump(model_pipeline, 'faculty_classifier_model.pkl')
    print("\nModel saved to 'faculty_classifier_model.pkl'")
