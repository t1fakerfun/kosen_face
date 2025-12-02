import cv2
import numpy as np
from tqdm import tqdm
import os

def detect_and_align_face(image_path, size=(224, 224)):
    """OpenCVを使った顔検出と整列"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            return None
        
        # 顔検出
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print(f"[WARN] No face detected: {image_path}")
            return None
        
        # 最大の顔を選択
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        
        # 顔領域を少し広めに切り出し
        padding_ratio = 0.3  # 30%の余白を追加
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)
        
        face_roi = image[y1:y2, x1:x2]
        face_gray = gray[y1:y2, x1:x2]
        
        # 目検出（整列用）
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        
        aligned_face = face_roi
        
        if len(eyes) >= 2:
            # 2つの目を見つけた場合、角度補正
            eyes = sorted(eyes, key=lambda eye: eye[0])  # x座標でソート
            
            # 最も離れている2つの目を選択（左右の目）
            if len(eyes) > 2:
                left_eye = eyes[0]
                right_eye = eyes[-1]
            else:
                left_eye = eyes[0]
                right_eye = eyes[1]
            
            # 目の中心を計算
            left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
            
            # 左右の目の距離が十分ある場合のみ回転補正
            eye_distance = np.sqrt((right_center[0] - left_center[0])**2 + (right_center[1] - left_center[1])**2)
            
            if eye_distance > 20:  # 最小距離チェック
                # 角度計算
                dx = right_center[0] - left_center[0]
                dy = right_center[1] - left_center[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # 回転の中心（顔ROIの中心）
                h_roi, w_roi = face_roi.shape[:2]
                center = (w_roi//2, h_roi//2)
                
                # 回転行列
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned_face = cv2.warpAffine(face_roi, M, (w_roi, h_roi))
                
                print(f"[INFO] Aligned face with {len(eyes)} eyes detected, angle: {angle:.1f}°")
            else:
                print(f"[INFO] Eyes too close, skipping rotation: {image_path}")
        else:
            print(f"[INFO] Eyes not detected clearly, using original: {image_path}")
        
        # リサイズ
        face = cv2.resize(aligned_face, size)
        
        return face
        
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return None

def enhance_image(image):
    """画像の前処理（コントラスト調整など）"""
    # ヒストグラム均一化
    if len(image.shape) == 3:
        # カラー画像の場合
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # グレースケール画像の場合
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
    
    return enhanced

def make_average_face(image_folder, output_path, enhance=True):
    """OpenCVのみで平均顔作成"""
    faces = []
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    print(f"Processing {len(files)} images with OpenCV...")
    
    for file in tqdm(files, desc="Processing"):
        path = os.path.join(image_folder, file)
        face = detect_and_align_face(path)
        
        if face is not None:
            if enhance:
                face = enhance_image(face)
            faces.append(face)
            print(f"[SUCCESS] {file}")
        else:
            print(f"[SKIP] {file}")
    
    if not faces:
        print("No faces detected.")
        return None
    
    print(f"\nSuccessfully processed {len(faces)}/{len(files)} images")
    
    # 平均顔計算（より正確な方法）
    faces_array = np.array(faces, dtype=np.float32)
    avg_face = np.mean(faces_array, axis=0).astype(np.uint8)
    
    # 保存
    cv2.imwrite(output_path, avg_face)
    print(f"Average face saved: {output_path}")
    
    # デバッグ用：処理された個別の顔も保存
    debug_folder = "processed_faces"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    
    for i, face in enumerate(faces[:10]):  # 最初の10枚を保存
        debug_path = os.path.join(debug_folder, f"processed_face_{i:02d}.jpg")
        cv2.imwrite(debug_path, face)
    
    print(f"Processed faces saved in {debug_folder}/")
    
    return avg_face

def create_comparison_image(faces, avg_face, output_path):
    """元の顔と平均顔の比較画像を作成"""
    if len(faces) == 0:
        return
    
    # 最初の4つの顔と平均顔を並べて表示
    comparison_faces = faces[:4] + [avg_face]
    
    # 横に並べて配置
    combined = np.hstack(comparison_faces)
    cv2.imwrite(output_path, combined)
    print(f"Comparison image saved: {output_path}")

if __name__ == "__main__":
    # 平均顔作成
    result = make_average_face("dataset/information_course", "avg_face_opencv.jpg")
    
    if result is not None:
        print("✓ Average face creation completed!")
        
        # 比較画像も作成
        faces = []
        files = os.listdir("processed_faces")[:4]
        for file in files:
            face = cv2.imread(f"processed_faces/{file}")
            if face is not None:
                faces.append(face)
        
        if faces:
            create_comparison_image(faces, result, "comparison.jpg")
    else:
        print("✗ Failed to create average face.")