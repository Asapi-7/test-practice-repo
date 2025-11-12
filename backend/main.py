import os
import uuid
import shutil
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio, time, shutil
from pydantic import BaseModel
from PIL import Image
import base64
from ml_model import load_ml_model, detect_face

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        return f"data:image/png;base64,{encoded_bytes.decode('utf-8')}"

ID_ACCESS_LOG = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # サーバー起動時にMLモデルをロードする
    load_ml_model()
    
    task = asyncio.create_task(cleanup_id())
    yield
    task.cancel()
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

# tempフォルダの画像を削除
async def cleanup_id():
    while True:
        print("cleanup now")
        now = time.time() # 1. 現在の時刻を取得
        
        # 2. アクセス履歴(ID_ACCESS_LOG)をチェック
        #    list()で囲むのは、ループ中に辞書を変更してもエラーにならないようにするため
        for upload_image_id, last_access in list(ID_ACCESS_LOG.items()):
            
            # 3. 最後のアクセスから600秒 (10分) 以上経過したかチェック
            if (now - last_access > 600):
                # 4. 10分以上経過していたら、対応する一時フォルダを削除
                dir_path = os.path.join(TEMP_DIR, upload_image_id)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path) # フォルダごと削除
                    del ID_ACCESS_LOG[upload_image_id] # アクセス履歴からも削除
                    print(f"ID: {upload_image_id} は古くなったので削除しました。")
        # --- ここまでがクリーンアップ処理 ---
        
        await asyncio.sleep(60)

# --- 1. FastAPIアプリの初期化 ---
app = FastAPI(lifespan=lifespan)

# --- 2. ディレクトリ設定 ---
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# フロントエンドの静的ファイルを置く場所
# 例: pet.html, EffectSelect.js, ImageDownload.js, ImageImport.js
WWW_DIR = "www"
os.makedirs(WWW_DIR, exist_ok=True)

# /static 配下で www/ のファイルを公開
# -> http://localhost:8000/static/pet.html で pet.html が見える
# -> http://localhost:8000/static/EffectSelect.js でJSが見える
app.mount("/static", StaticFiles(directory=WWW_DIR), name="static")

# =========================================================


# --- 3. ランドマークを保存するための仮のデータベース ---
image_landmark_storage: Dict[str, Dict] = {}

# #################################################
# ## ★★★ スタンプごとの設定データ（構成体） ★★★
# #################################################
STAMP_PLACEMENT_RULES = {
    "glasses_touka.png": {
        "type": "glasses",
        "required_landmarks": ["left_eye", "right_eye"]
    },
    "ribbon.png": {
        "type": "hat",
        "required_landmarks": ["forehead", "left_eye", "right_eye"]
    },
    "red_nose.png": {
        "type": "nose",
        "required_landmarks": ["nose"]
    }
}

# スタンプごとの「基準となる横幅(px)」
# この横幅を1.0倍として、顔に合わせてスケールを計算する
STAMP_BASE_WIDTHS = {
    "glasses_touka.png": 100,
    "ribbon.png": 120,
    "red_nose.png": 40,
    # 必要に応じて追加
}

# --- 4. データ形式の定義 (Pydanticモデル) --
class StampRequestData(BaseModel):
    upload_image_id: str
    stamp_id: str

# --- 5. ダミーのランドマーク検出関数 ---
def get_landmarks_from_face(image_path: str) -> Dict | None:
    """
    顔検出モデルを呼び出し、顔枠からランドマークを「推定」する関数
    """
    # 1. MLモデルで顔枠を検出
    face_data = detect_face(image_path)
    
    if face_data is None:
        print("❌ MLモデルが顔を検出できませんでした。")
        return None # 顔が見つからなかった
        
    bbox, score = face_data
    xmin, ymin, xmax, ymax = bbox
    
    print(f"✅ MLモデルが顔を検出しました: score={score:.2f}")

    # 2. 顔枠から各パーツの座標を「推定」する
    # (これはダミーロジックであり、より正確な推定ロジックに改善可能です)
    face_center_x = (xmin + xmax) / 2
    face_center_y = (ymin + ymax) / 2
    face_width = xmax - xmin
    face_height = ymax - ymin

    landmarks = {
        "left_eye": {
            "x": int(face_center_x - face_width * 0.2), 
            "y": int(face_center_y - face_height * 0.1)
        },
        "right_eye": {
            "x": int(face_center_x + face_width * 0.2), 
            "y": int(face_center_y - face_height * 0.1)
        },
        "nose": {
            "x": int(face_center_x), 
            "y": int(face_center_y + face_height * 0.1)
        },
        "forehead": {
            "x": int(face_center_x), 
            "y": int(ymin + face_height * 0.1) # 顔枠の上部付近
        }
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
    
    # MLモデル（顔枠）からランドマークを推定
    landmarks = get_landmarks_from_face(original_image_path)
    
    # もしMLが失敗したら、フォールバックとして従来のダミーロジックを使う
    if landmarks is None:
        print("ML検出失敗。ダミーロジックで代替します。")
        # 以前のダミー関数を呼び出す（関数名を変更）
        landmarks = detect_landmarks_dummy_fallback(original_image_path)
        if landmarks is None: # 万が一ダミーも失敗した場合
            raise HTTPException(status_code=400, detail="Failed to process image.")

    image_landmark_storage[upload_image_id] = landmarks
    
    ID_ACCESS_LOG[upload_image_id] = time.time()
    
    return JSONResponse(content={"upload_image_id": upload_image_id})

def detect_landmarks_dummy_fallback(image_path: str) -> Dict:
    """ML失敗時のフォールバック用ダミー関数（元コードと同じ）"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            center_x, center_y = width // 2, height // 2
            return {
                "left_eye": {"x": 150, "y": 230},
                "right_eye": {"x": 250, "y": 270},
                "nose": {"x": 200, "y": 210},
                "forehead": {"x": 200, "y": 120}
            }
    except Exception:
        return None

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
# 2) スタンプ画像ファイルを www/<stamp_id> から読む
    stamp_path = os.path.join(WWW_DIR, data.stamp_id + ".png")
    if not os.path.exists(stamp_path):
        raise HTTPException(
            status_code=404,
            detail=f"Stamp asset '{data.stamp_id}' not found on server."
        )

    # Base64化して、フロントが直接<img src="...">に使える形にする
    stamp_image_b64 = encode_image_to_base64(stamp_path)


    stamp_config = STAMP_PLACEMENT_RULES.get(data.stamp_id)

    if not stamp_config:
        nose_landmark = landmarks.get("nose", {"x": 100, "y": 100})
        return JSONResponse(content={
            "stamp_id": data.stamp_id, "x": nose_landmark["x"], "y": nose_landmark["y"], "scale": 1, "stamp_image": stamp_image_b64
        })

    stamp_type = stamp_config["type"]
    
    for required in stamp_config["required_landmarks"]:
        if required not in landmarks:
            raise HTTPException(status_code=400, detail=f"Required landmark '{required}' not found for this stamp.")

    x, y, needed_width_px = 0, 0, 100

    if stamp_type == "glasses":
        left_eye_landmark = landmarks["left_eye"]
        right_eye_landmark = landmarks["right_eye"]
        x = (left_eye_landmark["x"] + right_eye_landmark["x"]) // 2
        y = (left_eye_landmark["y"] + right_eye_landmark["y"]) // 2
        eye_dist = abs(right_eye_landmark["x"] - left_eye_landmark["x"])
        needed_width_px = eye_dist + 20  # 目の距離 + ちょい余白
    
    elif stamp_type == "hat":
        forehead_landmark = landmarks["forehead"]
        x, y = forehead_landmark["x"], forehead_landmark["y"]
        eye_dist = abs(landmarks["right_eye"]["x"] - landmarks["left_eye"]["x"])
        needed_width_px = eye_dist * 2  # 帽子は顔幅よりちょい大きく

    elif stamp_type == "nose":
        nose_landmark = landmarks["nose"]
        x, y = nose_landmark["x"], nose_landmark["y"]
        needed_width_px = 50  # 鼻スタンプは固定気味
    # 基準幅から倍率を計算
    base_width_px = STAMP_BASE_WIDTHS.get(data.stamp_id, 100)
    if base_width_px <= 0:
        base_width_px = 100
    scale = needed_width_px / base_width_px
    
    return JSONResponse(content={
        "stamp_id": data.stamp_id,
        "x": x,
        "y": y,
        # フロント側でスタンプ画像を何倍にすればいいか
        "scale": scale,
        "stamp_image": stamp_image_b64
        
    }
)
    # ======加えたよ===================================================
# 10. フロントファイルをアップロードするエンドポイント
#
# フロント班はここに pet.html / JSファイル をPOSTするだけでいい。
# そうするとサーバー側の www/ に保存される。
#
# curl例:
# curl -X POST "http://localhost:8000/upload_static_files" \
#   -F "files=@pet.html" \
#   -F "files=@EffectSelect.js" \
#   -F "files=@ImageDownload.js" \
#   -F "files=@ImageImport.js"
#
# 成功後は:
#   http://localhost:8000/static/pet.html
# でブラウザ表示できるようになる。
# =========================================================
@app.post("/upload_static_files", tags=["0. Frontend Static Upload"])
async def upload_static_files(files: List[UploadFile] = File(...)):
    
    
    saved_urls: List[str] = []

    for uploaded in files:
        # ファイル名だけ取り出す（"C:\\path\\to\\pet.html" みたいなのを防ぐ）
        filename = os.path.basename(uploaded.filename)

        if not filename:
            raise HTTPException(status_code=400, detail="File has no name")

        # 拡張子チェック：.html / .js 以外は拒否（安全のため）
        _, ext = os.path.splitext(filename.lower())
        if ext not in [".html", ".htm", ".js", ".css", ".png", ".jpg", ".jpeg", ".gif"]:
            raise HTTPException(
                status_code=400,
                detail=f"Extension not allowed: {ext}"
            )

        # 保存先は www/<filename>
        dest_path = os.path.join(WWW_DIR, filename)

        # アップロード内容を保存
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(uploaded.file, out_file)

        # 後で確認しやすいように、公開URLも返す
        saved_urls.append(f"/static/{filename}")
    return JSONResponse(content={
        "status": "ok",
        "uploaded_files": saved_urls,
        "hint": "ブラウザで /static/pet.html を開いて動作確認してください。"
    })
