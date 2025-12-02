from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import predict  # predict.py をインポート
import train    # train.py をインポート

app = FastAPI()

# ディレクトリ設定
DATASET_DIR = "dataset"
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

@app.post("/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    画像を受け取り、predict.pyのロジックで予測結果を返す
    """
    # 1. 画像を一時保存
    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. predict.py の関数を呼び出す
        # predict.py 内の関数名に合わせてください (predict_groupなど)
        label, probabilities = predict.predict_group(temp_path)

        # JSON互換にするため確率辞書を整形
        if probabilities:
            # numpy型などをfloatに変換
            probabilities = {k: float(v) for k, v in probabilities.items()}

        return {
            "status": "success",
            "predicted_label": label,
            "probabilities": probabilities
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # (任意) 一時ファイルの削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/register")
async def api_register(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    label: str = Form(...) # Flutterからテキストで送られるグループ名
):
    """
    画像とラベルを受け取り、データセットに保存して再学習を実行
    """
    # 1. 保存先フォルダの作成 (dataset/group_name/)
    save_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    # 2. 画像を保存
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. 学習を実行
    # 学習は時間がかかるため、レスポンスを先に返してバックグラウンドで実行するのが定石です。
    # すぐに結果を知りたい場合は background_tasks を使わず直接 train.run_training() を呼びます。

    # ここでは同期的に実行する例（学習が終わるまでFlutterが待機する）
    try:
        train.run_training() # train.pyを関数化したもの
        return {
            "status": "success",
            "message": f"Image saved to {label} and model retrained."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
