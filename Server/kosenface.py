# https://kiyosucyberclub.web.fc2.com/FaceRecognition/FaceR_V01-06.html
from deepface import DeepFace
import os

def find_similar_faces(upload_image_path, candidates_folder="candidates", top_k = 5,model_name = 'ArcFace'):
    df = DeepFace.find(img_path = upload_image_path, db_path = candidates_folder, model_name = model_name, enforce_detection=False)
    if df.empty:
        print("類似する顔が見つかりませんでした。")
        return []
    else:
        results = []
        df_sorted = df.sort_values(by=['VGG-Face_cosine'], ascending=True)
        for idx, row in df_sorted.head(top_k).iterrows():
            result.append({
                "path": row['identity'],
                "distance": row['VGG-Face_cosine']
            })
        return results

