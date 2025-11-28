import os
import torch
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Tuple, List, Any

# Faster R-CNNの依存関係
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from .resnet18_backbone import resnet18 # <-- これがあなたのresnet18_backbone.py

# UNetの依存関係
from .model import UNet
from . import config # config.IMAGE_SIZE, config.NUM_KEYPOINTS が定義されていることを想定

# 環境変数設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# デバイス設定
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ====================================================================
# モデル構築関数 (Faster R-CNNの新しい形式に合わせる)
# ====================================================================

def build_faster_rcnn_model(num_classes: int, weight_path: str, device: str):
    """Faster R-CNNモデルを構築し、重みをロードする"""
    try:
        # 1. バックボーン構築
        backbone = resnet18(pretrained=False)
        backbone.out_channels = 512 # ResNet18の場合、デフォルトは512 (torchvision標準ではない場合)
        
        # 2. アンカー生成器構築
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        # 3. Faster R-CNNモデル構築
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator
        )

        # 4. 重みロードと評価モード設定
        print(f"✅ Loading Faster R-CNN weights: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Faster R-CNNロードエラー: {e}")
        return None


def build_unet_model(num_keypoints: int, weight_path: str, device: str):
    """UNetモデルを構築し、重みをロードする"""
    try:
        model = UNet(in_channels=3, out_channels=num_keypoints).to(device)
        print(f"✅ Loading UNet weights: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ UNetロードエラー: {e}")
        return None

# ====================================================================
# グローバルモデルのロード
# ====================================================================

# Faster R-CNN
FASTER_RCNN_WEIGHTS = "fasterrcnn_resnet18_adam_20.pth" 
face_detector = build_faster_rcnn_model(num_classes=2, weight_path=FASTER_RCNN_WEIGHTS, device=DEVICE)

# UNet
UNET_WEIGHTS = "unet_epoch_22.pth"
landmark_detector = build_unet_model(
    num_keypoints=config.NUM_KEYPOINTS, 
    weight_path=UNET_WEIGHTS, 
    device=DEVICE
)

# UNet用前処理
UNET_TRANSFORM = T.Compose([
    T.Resize(config.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================================================================
# ユーティリティ関数
# ====================================================================

def decode_heatmaps_to_coordinates(heatmaps: torch.Tensor) -> np.ndarray:
    """ヒートマップから座標をデコードし、NumPy配列 [C, 2] で返す"""
    # config.IMAGE_SIZEは(H, W)を想定
    _, num_keypoints, H, W = heatmaps.shape
    flat_heatmaps = heatmaps.view(1, num_keypoints, -1)
    max_indices = torch.argmax(flat_heatmaps, dim=2)
    y_coords = max_indices // W
    x_coords = max_indices % W
    
    x_coords = (x_coords.float() + 0.5)
    y_coords = (y_coords.float() + 0.5)
    
    coordinates = torch.stack((x_coords, y_coords), dim=2).squeeze(0) # [C, 2]
    return coordinates.numpy()


# ====================================================================
# メイン推論関数（バックエンド連携用）
# ====================================================================

def detect_bbx_and_lndmk(image_path: str, score_threshold: float = 0.3) -> List[Any] | None:
    """
    Faster R-CNNで顔を検出し、その領域に対してUNetでランドマークを検出する。
    
    Returns:
        成功時: [[x_min, y_min], [x_max, y_max], [lx1, ly1], [lx2, ly2], ...]
        失敗時: None
    """
    if face_detector is None or landmark_detector is None:
        print("❌ 必要なモデルがロードされていません。")
        return None

    try:
        img_original = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return None
    
    W, H = img_original.size
    
    # --- STEP 1: Faster R-CNNによる顔検出 ---
    
    img_tensor = F.to_tensor(img_original).to(DEVICE)

    with torch.no_grad():
        outputs = face_detector([img_tensor])

    output = outputs[0]
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    
    if len(scores) == 0:
        print("⚠️ No detections found.")
        return None
    
    # 最もスコアの高いBBoxを1つだけ選ぶ
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]
    best_score = scores[best_idx]

    # **【ご要望の閾値処理】**
    if best_score < score_threshold:
        print(f"⚠️ No box above threshold ({best_score:.2f} < {score_threshold})")
        return None
    
    # BBox座標の抽出
    xmin, ymin, xmax, ymax = best_box.astype(float)
    
    # 座標を整数に丸める（クロップ用）
    xmin_int = max(0, int(xmin))
    ymin_int = max(0, int(ymin))
    xmax_int = min(W, int(xmax))
    ymax_int = min(H, int(ymax))

    # --- STEP 2: 検出された領域のクロップとUNet入力処理 ---
    
    cropped_img_pil = img_original.crop((xmin_int, ymin_int, xmax_int, ymax_int))
    crop_W, crop_H = cropped_img_pil.size
    
    if crop_W <= 0 or crop_H <= 0:
        print("⚠️ クロップされた領域のサイズが不正です。")
        return None
        
    cropped_img_tensor = UNET_TRANSFORM(cropped_img_pil).unsqueeze(0).to(DEVICE)
    
    # --- STEP 3: UNetによるランドマーク検出 ---
    
    with torch.no_grad():
        heatmaps_pred = landmark_detector(cropped_img_tensor).cpu()

    # ヒートマップから相対座標 [9, 2] を取得
    relative_landmarks_np = decode_heatmaps_to_coordinates(heatmaps_pred)

    # --- STEP 4: ランドマーク座標の絶対座標への変換 ---
    
    # 1. スケーリング係数の計算
    scale_x = crop_W / config.IMAGE_SIZE[1] 
    scale_y = crop_H / config.IMAGE_SIZE[0]
    
    # 2. スケーリング
    absolute_landmarks_np = relative_landmarks_np.copy()
    absolute_landmarks_np[:, 0] *= scale_x
    absolute_landmarks_np[:, 1] *= scale_y

    # 3. オフセットの適用 (元の画像上の絶対座標にする)
    absolute_landmarks_np[:, 0] += xmin_int
    absolute_landmarks_np[:, 1] += ymin_int

    # --- STEP 5: 最終的な出力形式の構築 ---
    
    # 検出されたバウンディングボックスの左上・右下座標
    output_list = [
        [float(xmin), float(ymin)], 
        [float(xmax), float(ymax)]
    ]
    
    # 9点のランドマーク座標を追加
    for x, y in absolute_landmarks_np:
        output_list.append([float(x), float(y)])
        
    print(f"✅ 検出完了。BBoxスコア: {best_score:.2f}")
    return output_list


# ====================================================================
# 実行部
# ====================================================================
if __name__ == "__main__":
    # Faster R-CNNの推論コードに合わせてテスト実行
    test_image = "test.jpg" # 適切な画像パスに変更
    SCORE_THRESHOLD = 0.3
    
    print("\n--- 推論開始 ---")
    result = detect_bbx_and_lndmk(test_image, score_threshold=SCORE_THRESHOLD) # test_imageに画像のパスを入れる当たり前だ！！！！！！！！！！
    
    if result is not None:
        print(f"要素数: {len(result)}")
        print(f"BBox (左上): {result[0]}")
        print(f"BBox (右下): {result[1]}")
        print(f"ランドマーク (最初の3点): {result}")
    else:
        print("❌ 処理失敗、または閾値未満の検出結果でした。")
