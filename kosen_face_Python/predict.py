#!/usr/bin/env python3

import pandas as pd
from deepface import DeepFace
import joblib
import warnings
import os
import glob

# DeepFaceのログを抑制 (任意ですが推奨)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

ACTIONS = ('age', 'gender', 'emotion', 'race')

# (別ファイルで実行する場合など) 保存したモデルを読み込む
try:
    model_pipeline = joblib.load('faculty_classifier_model.pkl')
    print("Model loaded successfully.")
except NameError:
    print("Model not found in memory. Please run Phase 1 first or load from file.")
    # exit() # スクリプトの場合は終了

def predict_group(new_image_path):
    """
    新しい顔写真を分析し、学習済みモデルでグループを予測する関数
    """
    try:
        # 1. DeepFaceで新しい画像を分析
        analysis_list = DeepFace.analyze(
            img_path=new_image_path,
            actions=ACTIONS,
            enforce_detection=False
        )

        # 2. 特徴量を抽出 (最初の顔)
        face_data = analysis_list[0]
        features_dict = {
            'age': face_data['age'],
            'gender': face_data['dominant_gender'],
            'emotion': face_data['dominant_emotion'],
            'race': face_data['dominant_race'],
        }

        # 3. モデルが学習したのと同じ形式 (DataFrame) に変換
        new_data_df = pd.DataFrame([features_dict])

        # 4. 予測の実行
        prediction = model_pipeline.predict(new_data_df)

        # 5. 各グループに属する「確率」も計算
        probabilities = model_pipeline.predict_proba(new_data_df)

        # 確率をラベルと紐付ける
        prob_dict = dict(zip(model_pipeline.classes_, probabilities[0]))

        return prediction[0], prob_dict

    except Exception as e:
        error_msg = str(e)
        if "Face could not be detected" in error_msg:
            return f"顔が検出できませんでした。画像に顔が写っているか確認してください。", None
        else:
            return f"Error analyzing image: {e}", None

if __name__ == "__main__":
    # --- 予測の実行例 ---
    # predict_imagesフォルダ内の全画像を取得
    PREDICT_DIR = "predict_images"
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(PREDICT_DIR, ext)))

    if not image_paths:
        print(f"No images found in {PREDICT_DIR} directory.")
    else:
        print(f"Found {len(image_paths)} image(s) to predict.\n")

        # 各画像に対して予測を実行
        for img_path in image_paths:
            predicted_label, probabilities = predict_group(img_path)

            if probabilities:
                print(f"\n--- Prediction for {img_path} ---")
                print(f"✅ Predicted Group: {predicted_label}")

                print("\nConfidence Scores:")
                for group, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
                    print(f"  - {group}: {prob*100:.2f}%")
                print("-" * 50)
            else:
                print(f"\n❌ Failed to predict {img_path}: {predicted_label}")
