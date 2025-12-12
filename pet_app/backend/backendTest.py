import os
import uuid
import shutil
import math # 角度計算用に追加しました（高井良）

# 書き加えた
from typing import Dict, List, Tuple
#

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
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

import json

#detect2.pyから機械学習モデルを読み込む
from .detect2 import load_ml_model, detect_face_and_lndmk

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
    "effectsangurasu": {"type": "glasses"},
    "effectsangurasu_migi": {"type": "glasses"},
    "effectsangurasu_hidari": {"type": "glasses"},
    "sangurasuA": {"type": "glasses"},
    "sangurasuA_migi": {"type": "glasses"},
    "sangurasuA_hidari": {"type": "glasses"},
    "sangurasuB": {"type": "glasses"},
    "sangurasuB_migi": {"type": "glasses"},
    "sangurasuB_hidari": {"type": "glasses"},
    "boushi":   { "type": "hat" },
    "santa":    { "type": "hat" },
    "fuwafuwa": { "type": "hat" },
    "effectribon": { "type": "kubi" },
    "nekutai":     { "type": "kubi" },
    "suzu":        { "type": "kubi" },
    "effecteye":        { "type": "eye" },
    "effecteye_katame": { "type": "eye" },
    "eye1":             { "type": "eye" },
    "eye1_migi":        { "type": "eye" },
    "eye1_hidari":      { "type": "eye" },
    "eye2":             { "type": "eye" },
    "eye2_katame":      { "type": "eye" },
    "effecthana": { "type": "hana" },
    "hige":       { "type": "hige" },
    "hige2":      { "type": "hige" },
    "effecthone": { "type": "kuchi" },
    "mouseA":     { "type": "kuchi" },
    "mouseB":     { "type": "kuchi" },
    "mimi":     { "type": "mimi" },
    "starmimi": { "type": "mimi" },
    "cat":      { "type": "mimi" },
    "effectA": { "type": "kira" },
    "effectB": { "type": "kira" },
    "effectC": { "type": "kira" }
}

# ちょうどいいスタンプのサイズを計算するために元画像の横幅のpxを設定しておく
STAMP_PX = {
    "effectsangurasu": 1052,
    "effectsangurasu_migi": 529,
    "effectsangurasu_hidari": 529,
    "sangurasuA":1000,
    "sangurasuA_migi": 498,
    "sangurasuA_hidari": 498,
    "sangurasuB":1000,
    "sangurasuB_migi": 495,
    "sangurasuB_hidari": 495,
    "boushi": 1000,
    "santa":1000,
    "fuwafuwa":1000,
    "effectribon": 904,
    "nekutai":396,
    "suzu":900,
    "effecteye": 978,
    "effecteye_katame": 305,
    "eye1":950,
    "eye1_migi": 269,
    "eye1_hidari": 269,
    "eye2":950,
    "eye2_katame": 344,
    "effecthana": 100,
    "hige":266,
    "hige2":155,
    "effecthone": 1024,
    "mouseA":1024,
    "mouseB":500,
    "mimi": 915,
    "starmimi":1000,
    "cat":900,
    "effectA":746,
    "effectB":694,
    "effectC":737
    }

# ユーザーからサーバーへのデータ形式を定義
class StampRequestData(BaseModel):
    upload_image_id: str
    stamp_id: str

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

# 画像からランドマークを検出する
def get_landmarks_from_face(image_path: str) -> Dict | None:

    result_data = detect_face_and_lndmk(image_path, score_threshold= 0.05)
    if result_data is None:
        print("❌ モデルが顔を検出できませんでした")
        return None, None, None
    
    result, score = result_data

    if len(result) < 11:
        print("⚠️ ランドマーク数が不足しています")
        return None, None, None
    
    # バウンディングボックス
    bbox_top_left = result[0]
    bbox_bottom_right = result[1]
    bbox = [bbox_top_left[0], bbox_top_left[1], bbox_bottom_right[0], bbox_bottom_right[1]]

    # ランドマーク
    raw_landmarks = result[2:11]

    print(f"✅ モデルが顔を検出し、ランドマークを計算しました。 スコア={score: .2f}")

    centers = get_center_landmarks(raw_landmarks, bbox)
    meta = {
        "raw_points": raw_landmarks,
        "bbox": bbox,
        "score": float(score)
    }
    return centers, meta, result

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
    bbox = meta.get("bbox")
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
        
        # バウンディングボックスがないとき用（高井良）
        x1 = int(face_cx - face_w / 2)
        y1 = int(eye_line_y - face_h * 0.4)
        x2 = int(x1 + face_w)
        y2 = int(y1 + face_h)

    # 横顔判定です。続きます。（高井良）
    # 目と目の距離
    eye_gap = math.sqrt((le["x"] - re["x"])**2 + (le["y"] - re["y"])**2)
    
    # 目と目の距離がbbox幅の 25% 未満なら横顔判定
    yokogao = (eye_gap / face_w) < 0.25

    # 角度計算
    if yokogao: # 横顔のときは顔の角度を0とみなす
        angle = 0.0
    else: # 正面顔のときは顔の傾きを算出する
        dy = le["y"] - re["y"] 
        dx = le["x"] - re["x"]
        angle = math.degrees(math.atan2(dy, dx))
    # 横顔判定ここまで（高井良）

    # -----------------------------
    # 2) スタンプ画像読み込み
    # -----------------------------

    # スタンプのタイプを取得（高井良）
    stamp_type = STAMP_PLACEMENT_RULES[data.stamp_id]["type"]
    filename = data.stamp_id

    # メガネと目については横顔の時片目用の画像を使う
    if yokogao and (data.stamp_id == "effectsangurasu" or data.stamp_id == "sangurasuA" or data.stamp_id == "sangurasuB" or data.stamp_id == "eye1"):
        if nose["x"] > face_cx: # 右向き
            filename = f"{data.stamp_id}_migi"
        else: # 左向き
            filename = f"{data.stamp_id}_hidari"
    
    if yokogao and (data.stamp_id == "effecteye" or data.stamp_id == "eye2"):
        filename = f"{data.stamp_id}_katame" # 右も左も同じ画像
    
    stamp_path = os.path.join(WWW_DIR, "effect/" + filename + ".png")
    if not os.path.exists(stamp_path):
        raise HTTPException(status_code=404, detail=f"スタンプ画像が見つかりません: {stamp_path}")

    # 元画像の横幅
    with Image.open(stamp_path) as s_img:
        original_w, original_h = s_img.size

        # 横顔の場合のスタンプ圧縮。続きます。（高井良）
        # スタンプごとに圧縮率を設定。
        compression_rules = {
            "mimi": 0.7, # 耳70%
            "kubi": 0.6, # リボン60%
            "kuchi": 0.7 # 口70%
        }

        # もし「横顔」かつ「設定リストにあるスタンプ」なら変形する
        if yokogao and stamp_type in compression_rules:
            
            # スタンプごとに圧縮率を取得
            rate = compression_rules[stamp_type]
            
            new_w = int(original_w * rate)
            new_h = original_h
            s_img = s_img.resize((new_w, new_h))
        
        # Base64変換
        buf = io.BytesIO()
        s_img.save(buf, format="PNG")
        stamp_image_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # サイズ情報を更新
        stamp_w, stamp_h = s_img.size
    # 横顔の場合のスタンプ圧縮ここまで（高井良）

    # 横顔のときの座標計算。続きます。左向き・右向き分岐以外は全部ゆめちゃんのコード使わせてもらいました。（高井良）
    if yokogao:
        # -----------------------------
        # 3) 基本量の計算
        # -----------------------------
        eye_dist = abs(re["x"] - le["x"])           # 目と目の距離
        eye_center_x = (le["x"] + re["x"]) / 2      # 目の中心X
        eye_center_y = (le["y"] + re["y"]) / 2      # 目の中心Y
        nose_mouth_dist = abs(mouth["y"] - nose["y"])

        # 生のランドマーク（9 点）も取り出しておく
        raw_points = meta.get("raw_points", None)

        # 初期値
        needed_width_px = eye_dist * 1.8   # だいたいいい感じの大きさ
        x_left = eye_center_x - needed_width_px/2
        y_top  = eye_center_y - needed_width_px/2

        if stamp_type == "glasses":
            if data.stamp_id == "effectsangurasu" or data.stamp_id == "sangurasuA" or data.stamp_id == "sungrasuB":
                if nose["x"] > face_cx: # 右向き
                    target_eye = le
                else:
                    target_eye = re # 左向き
                
                eye_center_x = target_eye["x"]
                eye_center_y = target_eye["y"]
                needed_width_px = face_w * 0.6
                aspect = stamp_h / stamp_w
                glasses_h_scaled = needed_width_px * aspect
                x_left = eye_center_x - needed_width_px / 2
                y_top  = eye_center_y - glasses_h_scaled / 2

            else:
                eye_center_x = (le["x"] + re["x"]) / 2
                eye_center_y = (le["y"] + re["y"]) / 2
                needed_width_px = face_w * 0.90
                aspect = stamp_h / stamp_w
                glasses_h_scaled = needed_width_px * aspect
                x_left = eye_center_x - needed_width_px / 2
                y_top  = eye_center_y - glasses_h_scaled / 2
        
        elif stamp_type == "eye":
            if data.stamp_id == "effecteye" or data.stamp_id == "eye1" or data.stamp_id == "eye2":
                if nose["x"] > face_cx: # 右向き
                    target_eye = le
                else:
                    target_eye = re # 左向き
                
                eye_center_x = (le["x"] + re["x"]) / 2
                eye_center_y = (le["y"] + re["y"]) / 2
                needed_width_px = face_w * 0.6
                aspect = stamp_h / stamp_w
                glasses_h_scaled = needed_width_px * aspect
                x_left = eye_center_x - needed_width_px / 2
                y_top  = eye_center_y - glasses_h_scaled / 2

            else:
                eye_center_x = (le["x"] + re["x"]) / 2
                eye_center_y = (le["y"] + re["y"]) / 2
                needed_width_px = face_w * 0.80
                aspect = stamp_h / stamp_w
                glasses_h_scaled = needed_width_px * aspect
                x_left = eye_center_x - needed_width_px / 2
                y_top  = eye_center_y - glasses_h_scaled / 2   

        elif stamp_type == "hat":
            # bx1, by1, bx2, by2 = bbox  # [xmin, ymin, xmax, ymax]
            # bbox_w = bx2 - bx1
            # bbox_h = by2 - by1
            # bbox_cx = (bx1 + bx2) / 2   # 横方向の中心
            # bbox_top_y = by1            # 上端の y（ここに帽子の底を合わせたい）
            
            # width_factor = 1.7         # 1.1 とかにすると少し大きくできる
            # needed_width_px = bbox_w * width_factor
            # aspect = stamp_h / stamp_w
            # hat_h_scaled = needed_width_px * aspect

            # if nose["x"] > face_cx: # 右向き
            #     OFFSET_X = 0
            #     x_left = bbox_cx - needed_width_px / 2
            # else: # 左向き
            #     OFFSET_X = 0
            #     x_left = bbox_cx - needed_width_px / 2
            # y_top = bbox_top_y - hat_h_scaled * 1.1
            raise HTTPException(status_code=400, detail="横顔には対応していません")

        elif stamp_type == "mimi":
            # bx1, by1, bx2, by2 = bbox  # [xmin, ymin, xmax, ymax]
            # bbox_w = bx2 - bx1
            # bbox_h = by2 - by1
            # bbox_cx = (bx1 + bx2) / 2   # 横方向の中心
            # bbox_top_y = by1            # 上端の y（ここに帽子の底を合わせたい）
            
            # width_factor = 1.7         # 1.1 とかにすると少し大きくできる
            # needed_width_px = bbox_w * width_factor
            # aspect = stamp_h / stamp_w
            # mimi_h_scaled = needed_width_px * aspect

            # if nose["x"] > face_cx: # 右向き
            #     OFFSET_X = 0
            #     x_left = bbox_cx - needed_width_px / 2
            # else: # 左向き
            #     OFFSET_X = 0
            #     x_left = bbox_cx - needed_width_px / 2
            # y_top = bbox_top_y - mimi_h_scaled * 0.8
            raise HTTPException(status_code=400, detail="横顔には対応していません")

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
        
        elif stamp_type == "hige":
            # 1. 0,1 点（bbox）から顔の横幅を計算 → face_w はすでに計算済み
            # 2. bbox の横幅に合わせてスタンプ画像をスケーリング
            needed_width_px = face_w * 0.80  # 鼻飾りなので少し小さめ（お好みで調整）

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

            if data.stamp_id == "mouseB":
                needed_width_px = face_w * 0.50
            # 3. スケーリング後の高さを計算
            aspect = stamp_h / stamp_w
            mouth_h_scaled = needed_width_px * aspect
            # 4. 9,10 点の中点に、スタンプ画像の中心が来るように配置
            if nose["x"] > face_cx: # 右向き
                offset_x = face_cx * 0.1        # 左右のズレが残るなら 0.02 * face_w とか入れて調整
                x_left = center_x - needed_width_px / 2 + offset_x
            else: # 左向き
                offset_x = face_cx * 0.1         # 左右のズレが残るなら 0.02 * face_w とか入れて調整
                x_left = center_x - needed_width_px / 2 - offset_x
            offset_y = face_h * 0.1
            y_top  = center_y - mouth_h_scaled / 2 + offset_y
        
        elif stamp_type == "kubi":
            # # bbox 底辺の中点を求める
            # bx1, by1, bx2, by2 = bbox
            # bbox_w = bx2 - bx1
            # center_bottom_x = (bx1 + bx2) / 2 
            # bottom_y = by2

            # # 1. 0と1点から bbox の横幅はすでに bbox_w で計算済み
            # # 2. bboxの横幅に合わせてスタンプ画像をスケーリング
            # width_factor = 1.0    # 顔幅のどれくらいにするか（0.4〜0.6で微調整）
            # needed_width_px = bbox_w * width_factor

            # # 3. スケーリングした画像の高さを取得
            # aspect = stamp_h / stamp_w
            # ribbon_h_scaled = needed_width_px * aspect

            # if nose["x"] > face_cx: # 右向き
            #     x_left = bx1
            # else: # 左向き
            #     x_left = bx2
            # y_top  = bottom_y
            raise HTTPException(status_code=400, detail="横顔には対応していません")
        
        elif stamp_type == "kira":
            original_image_path = os.path.join(
                TEMP_DIR, data.upload_image_id, "original.jpg"
                )
                
            with Image.open(original_image_path) as base_img:
                img_w, img_h = base_img.size  # ← 元画像の縦横
                
                needed_width_px = img_w
                x_left = 0
                y_top  = 0
                
                angle = 0.0
                rotation_center_x = 0.5
                rotation_center_y = 0.5    
            
        else: # その他のスタンプ
            needed_width_px = face_w * 0.5
            x_left = face_cx - needed_width_px / 2
            y_top = (y1 + y2)/2 - needed_width_px / 2
    # 横顔ここまで（高井良）
    
    # 正面顔のとき
    else:
        # -----------------------------
        # 3) 基本量の計算
        # -----------------------------
        eye_dist = abs(re["x"] - le["x"])           # 目と目の距離
        eye_center_x = (le["x"] + re["x"]) / 2      # 目の中心X
        eye_center_y = (le["y"] + re["y"]) / 2      # 目の中心Y
        nose_mouth_dist = abs(mouth["y"] - nose["y"])

        # 生のランドマーク（9 点）も取り出しておく
        raw_points = meta.get("raw_points", None)

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

        # 追加しました（高井良）
        elif stamp_type == "mimi":
            bx1, by1, bx2, by2 = bbox  # [xmin, ymin, xmax, ymax]
            bbox_w = bx2 - bx1
            bbox_h = by2 - by1
            bbox_cx = (bx1 + bx2) / 2   # 横方向の中心
            bbox_top_y = by1            # 上端の y（ここに帽子の底を合わせたい）
            
            width_factor = 1.0          # 1.1 とかにすると少し大きくできる
            needed_width_px = bbox_w * width_factor
            aspect = stamp_h / stamp_w
            mimi_h_scaled = needed_width_px * aspect
            OFFSET_X = 0
            OFFSET_Y = 0
            x_center = bbox_cx + OFFSET_X
            y_bottom = bbox_top_y + OFFSET_Y
            x_left = x_center - needed_width_px / 2
            y_top = y_bottom - (needed_width_px * aspect) # hatと違うのここだけです
        # ここまで（高井良）

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
        
        elif stamp_type == "hige":
            # 1. 0,1 点（bbox）から顔の横幅を計算 → face_w はすでに計算済み
            # 2. bbox の横幅に合わせてスタンプ画像をスケーリング
            needed_width_px = face_w * 0.80   # 鼻飾りなので少し小さめ（お好みで調整）

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

            if data.stamp_id == "mouseB":
                needed_width_px = face_w * 0.50

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
            original_image_path = os.path.join(
                TEMP_DIR, data.upload_image_id, "original.jpg"
                )
                
            with Image.open(original_image_path) as base_img:
                img_w, img_h = base_img.size  # ← 元画像の縦横
                
                needed_width_px = img_w
                x_left = 0
                y_top  = 0 
                
                angle = 0.0
                rotation_center_x = 0.5
                rotation_center_y = 0.5  

        else:
            # その他スタンプ（鼻あたり）
            needed_width_px = eye_dist * 1.0
            x_left = nose["x"] - needed_width_px/2
            y_top  = nose["y"] - needed_width_px/2

    # -----------------------------
    # 5) scale 計算（左上座標に丸め）
    # -----------------------------
    base_width_px = STAMP_PX.get(filename, stamp_w)
    if base_width_px <= 0:
        base_width_px = stamp_w
    
    if stamp_type == "kira":
        scale = max(img_w / stamp_w, img_h / stamp_h)
    else:
        scale = needed_width_px / base_width_px
    # x_int = int(round(x_left))
    # y_int = int(round(y_top))
    
    # 角度計算用に追加しました。続きます。（高井良）
    # 実際の表示サイズ（回転の中心計算に必要）
    final_display_w = stamp_w * scale
    final_display_h = stamp_h * scale
    
    # 回転軸の設定（0.5で中心）
    rotation_center_x = 0.5
    rotation_center_y = 0.5

    # スタンプごとの設定
    if stamp_type == "hat":
        rotation_center_y = 1.0 # 底辺中心

    elif stamp_type == "mimi":
        rotation_center_y = 1.0 # 底辺中心

    elif stamp_type == "kubi":
        rotation_center_y = 0.0 # 上辺中心

    # 回転軸が割合で計算されていたのをピクセルに変換することで描画できるようにする
    rotation_center_x_px = final_display_w * rotation_center_x
    rotation_center_y_px = final_display_h * rotation_center_y

    # 最終的な中心座標の計算
    final_center_x = x_left + rotation_center_x_px
    final_center_y = y_top + rotation_center_y_px

    # きらきらのエフェクトは回転させない
    if stamp_type == "kira":
        original_image_path = os.path.join(
            TEMP_DIR, data.upload_image_id, "original.jpg"
        )
        with Image.open(original_image_path) as base_img:
            img_w, img_h = base_img.size
            
        final_center_x = img_w / 2
        final_center_y = img_h / 2
        angle = 0.0   # 回転させない
    # 角度計算ここまで（高井良）

    # アクセスログ更新
    ID_ACCESS_LOG[data.upload_image_id] = time.time()

    return JSONResponse(content={
        "stamp_id": data.stamp_id,
        "x": int(final_center_x), # x_intから変えました
        "y": int(final_center_y), # y_intから変えました
        "scale": scale,
        "angle": angle,
        "rotation_center_x": rotation_center_x,
        "rotation_center_y": rotation_center_y,
        "stamp_image": stamp_image_b64
    })


# 接続状況確認用POST(三浦が追加)
@app.post("/check_ID", tags=["α. check_ID"])
async def check_ID(data: dict = Body(...)):
    userID = data.get("upload_image_id")
    user_ID_pass = os.path.join(TEMP_DIR, userID)
    if not os.path.exists(user_ID_pass):
        raise HTTPException(status_code=404, detail="長時間操作が無かったため、接続が切れました。再度画像を選択してください。")
    else:
        result = True

    return {"result": result}