import os
import uuid
import shutil

# 書き加えた
from typing import Dict, List, Tuple
#

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

# 勝手に足しました：みうら
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio, time, shutil, io
from backend.plot_results import plot_results
#

from pydantic import BaseModel
from PIL import Image

# 勝手に足しました：みうら
import base64
#

import re
import json

# モデルを切り替える
USE_NAKAYAMA_MODEL = False
USE_MORI_MODEL = True
USE_MIZUNUMA_MODEL = False

# モデル別のインポートと定義
if USE_NAKAYAMA_MODEL:
    from .ml_model import detect_face, load_ml_model

elif USE_MORI_MODEL:
    from .detect import load_ml_model, detect_face_and_lndmk

elif USE_MIZUNUMA_MODEL:
    from .mizunuma_model import load_ml_model, detect_face_and_lndmk

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        return f"data:image/png;base64,{encoded_bytes.decode('utf-8')}"

# 勝手に足しました：みうら
ID_ACCESS_LOG = {}
#

@asynccontextmanager
async def lifespan(app: FastAPI):
    # サーバー起動時にMLモデルをロードする

    load_ml_model()

    task = asyncio.create_task(cleanup_id())
    yield
    task.cancel()
    if os.path.exists(TEMP_DIR):   # 指定パスが存在するかを確かめる
        shutil.rmtree(TEMP_DIR)    # サーバーが閉じるとディレクトリを削除

# tempフォルダの画像を定期的に削除
async def cleanup_id():   # サーバーが開くと同時に１分おきの処理が始まる
    while True:
        print("cleanup now")
        now = time.time() # 現在の時刻を取得
        
        #  アクセス履歴(ID_ACCESS_LOG)をチェック
        for upload_image_id, last_access in list(ID_ACCESS_LOG.items()):
            
            # 最後のアクセスから10分以上経過したかチェック
            if (now - last_access > 600):
                # 10分以上経過していたら、対応するtempフォルダを削除
                dir_path = os.path.join(TEMP_DIR, upload_image_id)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path) # フォルダごと削除
                    del ID_ACCESS_LOG[upload_image_id] # アクセス履歴からも削除
                    print(f"ID: {upload_image_id} は古くなったので削除しました。")
        
        await asyncio.sleep(60)

# FastAPIアプリの初期化
app = FastAPI(lifespan=lifespan)

# アップロードされた画像を保存するためのtempディレクトリを作成
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# フロントエンドの静的ファイルを保存するためのwwwディレクトリを作成
# 例: pet.html, EffectSelect.js, ImageDownload.js, ImageImport.js
BASE_DIR = os.path.dirname(__file__)  # backendTest.py がある場所
WWW_DIR = os.path.join(BASE_DIR, "www")  # backend/www を指定

# /static 配下で www/ のファイルを公開 fastAPI動かす用
# -> http://localhost:8000/static/pet.html で pet.html が見える
# -> http://localhost:8000/static/EffectSelect.js でJSが見える
app.mount("/static", StaticFiles(directory=WWW_DIR), name="static")

# スタンプごとのタイプを設定
STAMP_PLACEMENT_RULES = {
    "boushi": {
        "type": "hat"
    },
    "effectatamaribon": {
        "type": "hat"
    },
    "effecthana": {
        "type": "hana"
    },
    "effecthone": {
        "type": "kuchi"
    },
    "effectsangurasu": {
        "type": "glasses"
    },
    "mimi": {
        "type": "hat"
    },
    "effecteye": {
        "type": "eye"
    },
    "effectribon": {
        "type": "kubi"
    },
    "kiraeffect": {
        "type": "kira"
    },
    "cat": {
        "type": "hat"
    },
    "eye1": {
        "type": "eye"
    },
    "eye2": {
        "type": "eye"
    },
    "glassguruguru": {
        "type": "glasses"
    },
    "hige": {
        "type": "kuchi"
    },
    "hoippu": {
        "type": "hat"
    },
    "nekonose": {
        "type": "hana"
    },
    "nekutai": {
        "type": "kubi"
    },
    "nezumi": {
        "type": "hat"
    },
    "santa": {
        "type": "hat"
    },
    "synglasshosi": {
        "type": "glasses"
    },
    "star": {
        "type": "hat"
    },
    "suzu": {
        "type": "kubi"
    },
    "tuno": {
        "type": "hat"
    }
}

# ちょうどいいスタンプのサイズを計算するために元画像の横幅のpxを設定しておく
# いらないかもこれ〜〜
STAMP_PX = {
    "boushi": 1000,
    "effecthana": 100,
    "effecthone": 1024,
    "effectribon": 904,
    "effectsangurasu": 1052,
    "mimi": 915,
    "effectatamaribon": 1112,
    "effecteye": 978,
    "kiraeffect":1536,
    "cat":900,
    "eye1":950,
    "eye2":950,
    "glassguruguru":1000,
    "hige":155,
    "hoippu":1000,
    "nekonose":266,
    "nekutai":396,
    "nezumi":900,
    "santa":1000,
    "star":1000,
    "sunglasshosi":1000,
    "suzu":900,
    "tuno":900
}



# ユーザーからサーバーへのデータ形式を定義
class StampRequestData(BaseModel):
    upload_image_id: str
    stamp_id: str

# あいちゃんのモデル
# ランドマーク９点の座標テキストデータをリストにする
def landmark_text_to_list(landmaek_text: str) -> List[List[float]]:
    points = []
    pattern = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')# 整数・小数・負の値を含む数値2つがある行を座標データとする

    # テキスト全体を走査
    for line in landmaek_text.splitlines():
        match = pattern.search(line.strip())
        if match :
            x = float(match.group(1))
            y = float(match.group(2))
            points.append([x, y])
    return points

# ランドマークから中心座標を計算する
# テキストデータはpoint0~8に行ごとに分かれてる。それぞれのpointに入ってる2つのデータに名前をつける。
def get_center_landmarks(points: List[List[float]]) -> Dict:
    right_eye_right_x, right_eye_right_y = points[0] # 右目右端
    right_eye_left_x, right_eye_left_y = points[1] # 右目左端
    left_eye_right_x, left_eye_right_y = points[2] # 左目右端
    left_eye_left_x, left_eye_left_y = points[3] # 左目左端
    nose_x, nose_y = points[4] # 鼻
    mouth_right_x, mouth_right_y = points[5] # 口右端
    mouth_left_x, mouth_left_y = points[6] # 口左端
    nose_to_mouth_x, nose_to_mouth_y = points[7] # 口上端
    mouth_center_x, mouth_center_y = points[8] # 口下端

    # 右目の中心座標を計算
    right_eye_x = (right_eye_right_x + right_eye_left_x) / 2
    right_eye_y = (right_eye_right_y + right_eye_left_y) / 2

    # 左目の中心座標を計算
    left_eye_x = (left_eye_right_x + left_eye_left_x) / 2
    left_eye_y = (left_eye_right_y + left_eye_left_y) / 2

    # 右目、左目、鼻、口をランドマーク辞書にする
    parts_landmarks = {
        "left_eye": { "x": int(left_eye_x), "y": int(left_eye_y)},
        "right_eye": { "x": int(right_eye_x), "y": int(right_eye_y)},
        "nose": { "x": int(nose_x), "y": int(nose_y)},
        "mouth": { "x": int(mouth_center_x), "y": int(mouth_center_y)}
    }
    return parts_landmarks

# あさひちゃんのモデル
# バウンディングボックスとランドマーク９点がリストで返ってくるので、それを使う
def get_center_landmarks(points: List[List[float]], bbox: List[float]) -> Dict:
    # 右目の中心座標を計算
    right_eye_x = (points[0][0] + points[1][0]) / 2
    right_eye_y = (points[0][1] + points[1][1]) / 2

    # 左目の中心座標を計算
    left_eye_x = (points[2][0] + points[3][0]) / 2
    left_eye_y = (points[2][1] + points[3][1]) / 2

    # 頭の中心座標を計算
    head_x = (bbox[0] + bbox[2]) / 2
    head_y = bbox[1] + ((bbox[3] - bbox[1]) / 2) #上辺のy座標+(縦幅/2)

    # 右目、左目、鼻、口、頭をランドマーク辞書にする
    parts_landmarks = {
        "left_eye": {"x": int(left_eye_x), "y": int(left_eye_y)},
        "right_eye": {"x": int(right_eye_x), "y": int(right_eye_y)},
        "nose": {"x": int(points[4][0]), "y": int(points[4][1])},
        "mouth": {"x": int(points[8][0]), "y": int(points[8][1])},
        "head": {"x": int(head_x), "y": int(head_y)}
    }
    return parts_landmarks

def detect_landmarks_text(image_path: str):
    # あいちゃんのモデル
    if USE_NAKAYAMA_MODEL:
        # 閾値0.05にした
        res = detect_face(image_path, threshold=0.05, allow_low_confidence=True)
        if not res:
            return None, None
        face_data = res
        # landmarks がないので bbox から近似ランドマークを作る
        try:
            bbox, score = face_data
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h * 0.45  # 鼻付近を中心に寄せる

            # 右目右端, 右目左端, 左目右端, 左目左端, 鼻, 口右, 口左, 鼻と口の間, 口中央
            approx = [
                [cx + w*0.20, cy - h*0.22],
                [cx + w*0.05, cy - h*0.22],
                [cx - w*0.05, cy - h*0.22],
                [cx - w*0.20, cy - h*0.22],
                [cx,             cy - h*0.05],
                [cx + w*0.15, cy + h*0.25],
                [cx - w*0.15, cy + h*0.25],
                [cx,             cy + h*0.12],
                [cx,             cy + h*0.25],
            ]
            return face_data, approx
        except Exception:
            return face_data, None
    
    # あさひちゃんとゆいちゃんのモデル
    elif USE_MORI_MODEL or USE_MIZUNUMA_MODEL:
        # 最初の２つはバウンディングボックスで残りはランドマーク[[xmin, ymin], [xmax, ymax], [lx1, ly1], ..., [lx9, ly9]]
        result_data = detect_face_and_lndmk(image_path, score_threshold=0.05) # 閾値0.05にしちゃった
        
        if result_data is None:
            print("⚠️ detect_face_and_lndmk が顔を検出できませんでした。")
            return None, None
        
        result, score = result_data
        
        if len(result) < 11:
            print("⚠️ ランドマーク数が不足しています。")
            return None, None
            
        # バウンディングボックス情報を抽出
        bbox_top_left = result[0]    # [xmin, ymin]
        bbox_bottom_right = result[1]  # [xmax, ymax]
        bbox = [bbox_top_left[0], bbox_top_left[1], bbox_bottom_right[0], bbox_bottom_right[1]]
        
        # ランドマーク9点を抽出
        landmarks = result[2:11]  # インデックス2から10までの9点
        
        face_data = (bbox, score)
        return face_data, landmarks, result     #resultを返さないとランドマーク表示ボタンが動かない、消しちゃダメ!
    
    else:
        print("モデルが選択されていません")
        return None, None


# 画像からランドマークを検出する
def get_landmarks_from_face(image_path: str) -> Dict | None:
    
    # モデルごとの処理分岐
    if USE_NAKAYAMA_MODEL:
        # MLモデルで顔枠を検出（返り値: (bbox, score), raw_landmark_text/list ）
        face_data, ML_LANDMARK_TEXT = detect_landmarks_text(image_path)
        
        # 顔検出ができなかった時
        if face_data is None:
            print("❌ MLモデルが顔を検出できませんでした。")
            return None, None
        
        # ランドマーク文字列またはリストから座標リストを作る
        face_landmarks_data = None
        if isinstance(ML_LANDMARK_TEXT, str):
            face_landmarks_data = landmark_text_to_list(ML_LANDMARK_TEXT)
        elif isinstance(ML_LANDMARK_TEXT, list):
            face_landmarks_data = ML_LANDMARK_TEXT
        else:
            print("❌ランドマーク情報が見つかりません。")
            return None, None

        if not face_landmarks_data:
            print("❌ランドマークのパースに失敗しました。")
            return None, None

        if len(face_landmarks_data) != 9:
            print(f"❌ランドマークは９点必要です。検出数: {len(face_landmarks_data)}")
            return None, None

        centers = get_center_landmarks(face_landmarks_data)
        
        # bboxとscoreを取得
        try:
            bbox, score = face_data
            print(f"✅MLモデルが顔を検出し、ランドマークを計算しました。score={score: .2f}")
        except Exception:
            bbox, score = None, None
    
    elif USE_MORI_MODEL or USE_MIZUNUMA_MODEL:
        face_data, face_landmarks_data, result = detect_landmarks_text(image_path)     #resultを得ないとランドマーク表示ボタンが動かない、消しちゃダメ!
        
        # 顔検出ができなかった時
        if face_data is None or face_landmarks_data is None:
            print("❌ MLモデルが顔を検出できませんでした。")
            return None, None
        
        # bboxとscoreを取得
        bbox, score = face_data
        print(f"✅MLモデルが顔を検出し、ランドマークを計算しました。score={score: .2f}")

        # bboxを渡してランドマーク辞書を作成
        centers = get_center_landmarks(face_landmarks_data, bbox)
    
    else:
        print("❌ モデルが選択されていません。")
        return None, None

    meta = {
        "raw_points": face_landmarks_data,
        "bbox": bbox,
        "score": score
    }
    # 戻り値は(centers, meta)
    return centers, meta, result     #resultを返さないとランドマーク表示ボタンが動かない、消しちゃダメ!

    # 顔検出はできたけど、ランドマークのテキストデータがおかしい時
    # 顔検出はできたけど、ランドマーク数が足りない時
    #if len(face_landmarks_data) != 9:
        #print(f"❌ランドマークは９点必要です。検出数: {len(face_landmarks_data)}")
        #return None
    
    #parts_landmarks = get_center_landmarks(face_landmarks_data)

    # 顔検出のスコアを計算
    #_, score = face_data
    #print(f"✅MLモデルが顔を検出し、ランドマークを計算しました。score={score: .2f}")
    #return parts_landmarks

# APIエンドポイントの作成
@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

# 画像アップロードとランドマーク処理(担当：高井良)
@app.post("/upload_and_detect", tags=["1. Image Upload & Landmark Detection"])
async def upload_and_detect_landmarks(file: UploadFile = File(...)):
    api_start_time = time.time()
    
    upload_image_id = str(uuid.uuid4())
    upload_temp_dir = os.path.join(TEMP_DIR, upload_image_id)
    os.makedirs(upload_temp_dir)
    original_image_path = os.path.join(upload_temp_dir, "original.jpg")
    
    # ファイルを保存
    with open(original_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # MLの推論時間を表示
    ml_start_time = time.time()
    centers, meta, result = get_landmarks_from_face(original_image_path)     #resultを得ないとランドマーク表示ボタンが動かない、消しちゃダメ!
    ml_end_time = time.time()
    print(f"ML推論時間: {(ml_end_time - ml_start_time) * 1000:.2f}ms")
    
    # もしMLが失敗したらエラー表示
    if centers is None:
        raise HTTPException(status_code=400, detail="ML検出失敗")
    
    # centersとmetaをtemp/<upload_id>/landmarks.jsonに保存
    landmarks_unity = {"centers": centers, "meta": meta}
    landmarks_JSON_path = os.path.join(upload_temp_dir, "landmarks.json")
    with open(landmarks_JSON_path, "w", encoding="utf-8") as f:
        json.dump(landmarks_unity, f, ensure_ascii=False)
    
    # 勝手に足しました:みうら
    ID_ACCESS_LOG[upload_image_id] = time.time() # アクセス履歴を残す

    landmark_plot = plot_results(original_image_path, result)
    buf = io.BytesIO()                      #データ変換の保存先生成
    landmark_plot.save(buf, format="PNG")   #保存
    buf.seek(0)                             #保存終わったから先頭に戻す(フィルムを先頭に戻す感じ？)
    landmark_plot_b64 = base64.b64encode(buf.getvalue()).decode()
    landmark_plot_uri = f"data:image/png;base64,{landmark_plot_b64}"

    return JSONResponse(content={
        "upload_image_id": upload_image_id,
        "landmark_plot": landmark_plot_uri})


# スタンプ情報の取得(担当：西本)
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
    # temp/<upload_image_id>/landmarks.jsonからランドマークを取得
    landmarks_unity = os.path.join(TEMP_DIR, data.upload_image_id, "landmarks.json")
    if not os.path.exists(landmarks_unity):
        raise HTTPException(status_code=404, detail="長時間操作が無かったため、接続が切れました。再度画像を選択してください。")
    with open(landmarks_unity, "r", encoding="utf-8") as f:
        unity = json.load(f)
    centers = unity.get("centers")
    meta = unity.get("meta")

    # centers = { left_eye:{x,y}, right_eye:{x,y}, nose:{x,y}, mouth:{x,y} }
    try:
        le = centers["left_eye"]
        re = centers["right_eye"]
        nose = centers["nose"]
        mouth = centers["mouth"]
        head = centers["head"] # 追加しました。あさひちゃんのモデルで使えます。（高井良）
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"必要ランドマーク不足: {e}")
        
        # ★ ここを追加（bbox の情報を取り出しておく）★
    bbox = meta.get("bbox")  # [x1, y1, x2, y2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) / 2

        # 「目の高さ」を顔の上からだいたい 38% あたりとみなす
        eye_line_y = y1 + face_h * 0.38
    else:
        # 万一 bbox が無ければ、従来通り目のランドマークなどでざっくり代用
        eye_dist   = abs(re["x"] - le["x"])
        face_w     = eye_dist * 2.0
        face_h     = face_w * 1.2
        face_cx    = (le["x"] + re["x"]) / 2
        eye_line_y = (le["y"] + re["y"]) / 2
        x1, y1 = int(face_cx - face_w / 2), int(eye_line_y - face_h * 0.4)
    # -----------------------------
    # 2) スタンプ画像読み込み
    # -----------------------------

# 2) スタンプ画像ファイルを www/<stamp_id> から読む!
    stamp_path = os.path.join(WWW_DIR, "effect/" + data.stamp_id + ".png")
    if not os.path.exists(stamp_path):
        raise HTTPException(status_code=404, detail=f"スタンプ画像が見つかりません: {stamp_path}")

    stamp_image_b64 = encode_image_to_base64(stamp_path)

    # 元画像の横幅
    with Image.open(stamp_path) as s_img:
        stamp_w, stamp_h = s_img.size

    # -----------------------------
    # 3) 基本量の計算
    # -----------------------------
    eye_dist = abs(re["x"] - le["x"])           # 目と目の距離
    eye_center_x = (le["x"] + re["x"]) / 2      # 目の中心X
    eye_center_y = (le["y"] + re["y"]) / 2      # 目の中心Y
    nose_mouth_dist = abs(mouth["y"] - nose["y"])

    # 生のランドマーク（9 点）も取り出しておく
    raw_points = meta.get("raw_points", None)
    # -----------------------------
    # 4) スタンプ種別ごとに位置とサイズを計算
    # -----------------------------
    stamp_type = STAMP_PLACEMENT_RULES[data.stamp_id]["type"]

    # 初期値
    needed_width_px = eye_dist * 1.8   # だいたいいい感じの大きさ
    x_left = eye_center_x - needed_width_px/2
    y_top  = eye_center_y - needed_width_px/2
    
    if stamp_type == "glasses":
        eye_center_x = (le["x"] + re["x"]) / 2
        eye_center_y = (le["y"] + re["y"]) / 2
        needed_width_px = face_w * 0.90
        aspect = stamp_h / stamp_w
        glasses_h_scaled = needed_width_px * aspect
        x_left = eye_center_x - needed_width_px / 2
        y_top  = eye_center_y - glasses_h_scaled / 2

    elif stamp_type == "eye":
        eye_center_x = (le["x"] + re["x"]) / 2
        eye_center_y = (le["y"] + re["y"]) / 2
        needed_width_px = face_w * 0.80
        aspect = stamp_h / stamp_w
        glasses_h_scaled = needed_width_px * aspect
        x_left = eye_center_x - needed_width_px / 2
        y_top  = eye_center_y - glasses_h_scaled / 2
        
    elif stamp_type == "hat":
        bx1, by1, bx2, by2 = bbox  # [xmin, ymin, xmax, ymax]
        bbox_w = bx2 - bx1
        bbox_h = by2 - by1
        bbox_cx = (bx1 + bx2) / 2   # 横方向の中心
        bbox_top_y = by1            # 上端の y（ここに帽子の底を合わせたい）
            
        width_factor = 1.0          # 1.1 とかにすると少し大きくできる
        needed_width_px = bbox_w * width_factor
        aspect = stamp_h / stamp_w
        hat_h_scaled = needed_width_px * aspect
        OFFSET_X = 0
        OFFSET_Y = 0
        x_center = bbox_cx + OFFSET_X
        y_bottom = bbox_top_y + OFFSET_Y
        x_left = x_center - needed_width_px / 2 
        y_top  = y_bottom - hat_h_scaled

    elif stamp_type == "gantai":
        # ● 眼帯（左目用）：顔の左寄りの目あたりに置く
        needed_width_px = face_w * 0.60
        aspect = stamp_h / stamp_w
        patch_h_scaled = needed_width_px * aspect
        left_eye_cx = re["x"]
        left_eye_cy = re["y"]
        x_left = left_eye_cx - needed_width_px / 2
        y_top  = left_eye_cy - patch_h_scaled / 2
        
    # ⑥ 鼻の飾り
    elif stamp_type == "hana":
        # 1. 0,1 点（bbox）から顔の横幅を計算 → face_w はすでに計算済み
        # 2. bbox の横幅に合わせてスタンプ画像をスケーリング
        needed_width_px = face_w * 0.28   # 鼻飾りなので少し小さめ（お好みで調整）

        # 3. スケーリング後の高さを計算
        aspect = stamp_h / stamp_w
        nose_h_scaled = needed_width_px * aspect

        # 4. ランドマーク 6 点（centers["nose"]）の位置に
        #    スタンプ画像の「中心」が来るように配置
        center_x = nose["x"]
        center_y = nose["y"]

        x_left = center_x - needed_width_px / 2
        y_top  = center_y - nose_h_scaled / 2

    # ⑤ 口の飾り（ひげ・骨など）
    elif stamp_type == "kuchi":
        if isinstance(raw_points, list) and len(raw_points) >= 9:
            mouth_left  = raw_points[5]
            mouth_right = raw_points[6]
            mouth_up    = raw_points[7]
            mouth_down  = raw_points[8]
            
            center_x = (mouth_left[0] + mouth_right[0]) / 2.0
            center_y = (mouth_up[1]   + mouth_down[1]) / 2.0
        else:
            # 万一 raw_points が無い場合は、従来どおり mouth 中心を使う
            center_x = mouth["x"]
            center_y = mouth["y"]

        # 2. スタンプ画像をスケーリング（スケーリング方法はおまかせでよいとのことなので、
        #    顔幅の 30% くらいに設定）
        needed_width_px = face_w * 0.80

        # 3. スケーリング後の高さを計算
        aspect = stamp_h / stamp_w
        mouth_h_scaled = needed_width_px * aspect

        offset_x = 0.0         # 左右のズレが残るなら 0.02 * face_w とか入れて調整
        offset_y = 0.0
        # 4. 9,10 点の中点に、スタンプ画像の中心が来るように配置
        x_left = center_x - needed_width_px / 2 + offset_x
        y_top  = center_y - mouth_h_scaled / 2 + offset_y

    elif stamp_type == "kubi":
        # bbox 底辺の中点を求める
        bx1, by1, bx2, by2 = bbox 
        bbox_w = bx2 - bx1
        center_bottom_x = (bx1 + bx2) / 2 
        bottom_y = by2

        # 1. 0と1点から bbox の横幅はすでに bbox_w で計算済み
        # 2. bboxの横幅に合わせてスタンプ画像をスケーリング
        width_factor = 0.5    # 顔幅のどれくらいにするか（0.4〜0.6で微調整）
        needed_width_px = bbox_w * width_factor

        # 3. スケーリングした画像の高さを取得
        aspect = stamp_h / stamp_w
        ribbon_h_scaled = needed_width_px * aspect

        # 4. bbox の下側の中点と、スタンプ画像の「上側の中点」が一致するように配置
        #    → 上側の中点 = (x_left + needed_width_px/2, y_top)
        #       これを (center_bottom_x, bottom_y) に合わせる
        x_left = center_bottom_x - needed_width_px / 2
        y_top  = bottom_y

    elif stamp_type == "kira":
        # ★ ランドマーク・bbox を使って「顔のだいぶ周りまで」覆うエフェクトにする

        # 1. 顔の bbox 情報（他のスタンプでも使っているやつ）
        bx1, by1, bx2, by2 = bbox  # [xmin, ymin, xmax, ymax]
        face_w = bx2 - bx1
        face_h = by2 - by1

        # 2. 顔の中心座標
        face_cx = (bx1 + bx2) / 2
        face_cy = (by1 + by2) / 2

        # 3. 「顔の長いほうの辺」の 2.5 倍ぐらいに広げて、周りも覆うようにする
        face_long = max(face_w, face_h)
        needed_width_px = face_long * 2.0

        # 4. スタンプ画像の縦横比に合わせて高さを決める
        aspect = stamp_h / stamp_w
        kira_h_scaled = needed_width_px * aspect

        # 5. 顔の中心にスタンプの中心が来るように、左上座標を決める
        x_left = face_cx - needed_width_px / 2
        y_top  = face_cy - kira_h_scaled / 2

    else:
        # その他スタンプ（鼻あたり）
        needed_width_px = eye_dist * 1.0
        x_left = nose["x"] - needed_width_px/2
        y_top  = nose["y"] - needed_width_px/2

    # -----------------------------
    # 5) scale 計算（左上座標に丸め）
    # -----------------------------
    base_width_px = STAMP_PX.get(data.stamp_id, stamp_w)
    if base_width_px <= 0:
        base_width_px = stamp_w

    scale = needed_width_px / base_width_px
    x_int = int(round(x_left))
    y_int = int(round(y_top))

    # アクセスログ更新
    ID_ACCESS_LOG[data.upload_image_id] = time.time()

    return JSONResponse(content={
        "stamp_id": data.stamp_id,
        "x": x_int,
        "y": y_int,
        "scale": scale,
        "stamp_image": stamp_image_b64
    })
    
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