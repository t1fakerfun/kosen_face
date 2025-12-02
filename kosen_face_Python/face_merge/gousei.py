import cv2
import numpy as np
import face_alignment
from tqdm import tqdm
import os
import torch

# 正しい初期化方法
torch.cuda.is_available = lambda : False
device = torch.device("cpu")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

def align_face(image_path, size=(224, 224)):
    """顔のランドマークを使って顔を整列"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            return None
        
        # RGB形式に変換（face_alignmentはRGBを期待）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ランドマーク検出
        preds = fa.get_landmarks(image_rgb)
        if preds is None or len(preds) == 0:
            print(f"[WARN] No landmarks detected: {image_path}")
            return None
        
        landmarks = preds[0]  # 最初の顔を使用
        
        # 目の座標を取得（68点ランドマーク）
        left_eye = np.mean(landmarks[36:42], axis=0)   # 左目
        right_eye = np.mean(landmarks[42:48], axis=0)  # 右目
        
        # 目の角度を計算
        dx, dy = right_eye - left_eye
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 目の中心を計算
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # 回転行列を作成
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        # 画像を回転
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # 顔領域を切り出し（大まかな位置）
        face_width = int(np.linalg.norm(right_eye - left_eye) * 3)
        face_height = int(face_width * 1.3)
        
        center_x, center_y = int(eye_center[0]), int(eye_center[1])
        x1 = max(0, center_x - face_width // 2)
        y1 = max(0, center_y - face_height // 2)
        x2 = min(aligned.shape[1], center_x + face_width // 2)
        y2 = min(aligned.shape[0], center_y + face_height // 2)
        
        face_crop = aligned[y1:y2, x1:x2]
        
        # 指定サイズにリサイズ
        face = cv2.resize(face_crop, size)
        
        return face
        
    except Exception as e:
        print(f"[ERROR] Error processing {image_path}: {e}")
        return None

def align_face_simple(image_path, size=(224, 224)):
    """シンプルな顔検出とアライメント（代替手法）"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # OpenCVの顔検出
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # 最大の顔を選択
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            
            # 顔領域を少し広めに切り出し
            padding = int(w * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, size)
            return face
            
    except Exception as e:
        print(f"[ERROR] Simple alignment failed for {image_path}: {e}")
    return None

def make_average_face(image_folder, output_path, use_landmarks=True):
    """平均顔を作成"""
    faces = []
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    print(f"Processing {len(files)} images from {image_folder}...")
    
    for file in tqdm(files, desc="Processing images"):
        path = os.path.join(image_folder, file)
        
        # ランドマークベースの処理を試す
        face = None
        if use_landmarks:
            face = align_face(path)
        
        # 失敗した場合はシンプルな方法を試す
        if face is None:
            print(f"[INFO] Trying simple detection for {file}")
            face = align_face_simple(path)
        
        if face is not None:
            faces.append(face)
            print(f"[SUCCESS] Processed {file}")
        else:
            print(f"[SKIP] Could not process {file}")
    
    if not faces:
        print("No faces processed successfully.")
        return None
    
    print(f"\nSuccessfully processed {len(faces)} out of {len(files)} images")
    
    # 平均顔を計算
    avg = np.mean(faces, axis=0).astype(np.uint8)
    
    # 保存
    cv2.imwrite(output_path, avg)
    print(f"Average face saved: {output_path}")
    
    # 個別の顔も保存（デバッグ用）
    debug_folder = "debug_faces"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    
    for i, face in enumerate(faces[:5]):  # 最初の5枚だけ保存
        debug_path = os.path.join(debug_folder, f"aligned_face_{i}.jpg")
        cv2.imwrite(debug_path, face)
    
    print(f"Debug faces saved in {debug_folder}/")
    return avg

def check_installation():
    """必要なライブラリの動作確認"""
    try:
        import face_alignment
        print("✓ face_alignment library is available")
        
        # LandmarksTypeの利用可能な属性を確認
        print("Available LandmarksType attributes:")
        for attr in dir(face_alignment.LandmarksType):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        
        return True
    except ImportError as e:
        print(f"✗ face_alignment not installed: {e}")
        return False

# 使用例
if __name__ == "__main__":
    # ライブラリの確認
    if not check_installation():
        print("Please install face_alignment: pip install face_alignment")
        exit(1)
    
    # 平均顔を作成
    result = make_average_face("dataset/information_course", "avg_information_aligned.jpg")
    
    if result is not None:
        print("Average face creation completed!")
    else:
        print("Failed to create average face.")
