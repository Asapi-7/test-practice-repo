import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
from PIL import Image

# --- 1. FastAPIアプリの初期化 ---
app = FastAPI()

# --- 2. ディレクトリ設定 ---
STAMPS_DIR = "stamps"
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
os.makedirs(STAMPS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. データ形式の定義 (Pydanticモデル) ---
class StampInfo(BaseModel):
    stamp_id: str
    x: int
    y: int
    size: int
    angle: float = 0.0

class ApplyStampsData(BaseModel):
    session_id: str
    stamps: List[StampInfo]

# --- 4. APIエンドポイントの作成 ---

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

# #################################################
# ## 画像合成機能
# #################################################

# --- 変更 ---
# 機能3: スタンプを合成するAPIから、実際の合成処理を削除しました
@app.post("/apply_stamps", tags=["Feature 3: Apply Multiple Stamps"])
async def apply_stamps_endpoint(data: ApplyStampsData):
    """
    元画像の存在確認のみを行い、実際の合成処理は行いません。
    """
    original_image_path = os.path.join(TEMP_DIR, data.session_id, "original.jpg")
    if not os.path.exists(original_image_path):
        raise HTTPException(status_code=404, detail="Session ID not found.")

    # 処理が成功したことだけを伝えるメッセージを返します
    return JSONResponse(content={"message": "Request received, but image synthesis is disabled."})

# 合成の元となる画像をアップロードするAPI (変更なし)
@app.post("/upload", tags=["Base Function"])
async def upload_image(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_temp_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_temp_dir)
    original_image_path = os.path.join(session_temp_dir, "original.jpg")
    with open(original_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return JSONResponse(content={"session_id": session_id})
