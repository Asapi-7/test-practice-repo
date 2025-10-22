import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Dict
from PIL import Image

# --- 1. FastAPIアプリの初期化 ---
app = FastAPI()

# --- 2. ディレクトリ設定 ---
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- 3. ランドマークを保存するための仮のデータベース ---
image_landmark_storage: Dict[str, Dict] = {}

# #################################################
# ## ★★★ スタンプごとの設定データ（構成体） ★★★
# #################################################
STAMP_PLACEMENT_RULES = {
    "glasses.png": {
        "type": "glasses",
        "required_landmarks": ["left_eye", "right_eye"]
    },
    "party_hat.png": {
        "type": "hat",
        "required_landmarks": ["forehead", "left_eye", "right_eye"]
    },
    "clown_nose.png": {
        "type": "nose",
        "required_landmarks": ["nose"]
    }
}

# --- 4. データ形式の定義 (Pydanticモデル) --
class StampRequestData(BaseModel):
    upload_image_id: str
    stamp_id: str

# --- 5. ダミーのランドマーク検出関数 ---
def detect_landmarks_dummy(image_path: str) -> Dict:
    with Image.open(image_path) as img:
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        landmarks = {
            "left_eye": {"x": center_x - 50, "y": center_y - 30},
            "right_eye": {"x": center_x + 50, "y": center_y - 30},
            "nose": {"x": center_x, "y": center_y + 10},
            "forehead": {"x": center_x, "y": center_y - 80}
        }
        return landmarks

# --- 6. APIエンドポイントの作成 ---
@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

# --- 機能①：画像アップロードとランドマーク処理 ---
@app.post("/upload_and_detect", tags=["1. Image Upload & Landmark Detection"])
async def upload_and_detect_landmarks(file: UploadFile = File(...)):
    upload_image_id = str(uuid.uuid4())
    upload_temp_dir = os.path.join(TEMP_DIR, upload_image_id)
    os.makedirs(upload_temp_dir)
    original_image_path = os.path.join(upload_temp_dir, "original.jpg")
    
    with open(original_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    landmarks = detect_landmarks_dummy(original_image_path)
    
    image_landmark_storage[upload_image_id] = landmarks
    
    return JSONResponse(content={"upload_image_id": upload_image_id})

# #################################################
# ## 機能②：スタンプ情報の取得 (ロジック変更)
# #################################################
@app.post("/get_stamp_info", tags=["2. Get Stamp Info"])
# ★★★ 引数名と型を元に戻しました ★★★
async def get_stamp_info(data: StampRequestData):
    """
    ② ユーザーが加工したいスタンプの名前とIDを受け取る
    ↓
    　 IDからランドマークを取得して、適切な座標を取得する
    ↓
    　 スタンプの名前とエフェクトを貼る位置とサイズを返す
    """

    landmarks = image_landmark_storage.get(data.upload_image_id)
    if not landmarks:
        raise HTTPException(status_code=404, detail="Upload Image ID not found.")

    stamp_config = STAMP_PLACEMENT_RULES.get(data.stamp_id)

    if not stamp_config:
        nose_landmark = landmarks.get("nose", {"x": 100, "y": 100})
        return JSONResponse(content={
            "stamp_id": data.stamp_id, "x": nose_landmark["x"], "y": nose_landmark["y"], "size": 100
        })

    stamp_type = stamp_config["type"]
    
    for required in stamp_config["required_landmarks"]:
        if required not in landmarks:
            raise HTTPException(status_code=400, detail=f"Required landmark '{required}' not found for this stamp.")

    x, y, size = 0, 0, 100

    if stamp_type == "glasses":
        left_eye_landmark = landmarks["left_eye"]
        right_eye_landmark = landmarks["right_eye"]
        x = (left_eye_landmark["x"] + right_eye_landmark["x"]) // 2
        y = (left_eye_landmark["y"] + right_eye_landmark["y"]) // 2
        size = abs(right_eye_landmark["x"] - left_eye_landmark["x"]) + 20
    
    elif stamp_type == "hat":
        forehead_landmark = landmarks["forehead"]
        x, y = forehead_landmark["x"], forehead_landmark["y"]
        size = abs(landmarks["right_eye"]["x"] - landmarks["left_eye"]["x"]) * 2

    elif stamp_type == "nose":
        nose_landmark = landmarks["nose"]
        x, y = nose_landmark["x"], nose_landmark["y"]
        size = 50
        
    return JSONResponse(content={
        "stamp_id": data.stamp_id,
        "x": x,
        "y": y,
        "size": size
    })
