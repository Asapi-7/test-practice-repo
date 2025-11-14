import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from PIL import Image
from torchvision import transforms as T
from resnet18_backbone import resnet18 # 注: このファイルも同じフォルダに必要です
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import RetinaNet
from typing import Tuple, List, Optional

# --- グローバル変数としてモデルとデバイスを保持 ---
model: Optional[RetinaNet] = None
device: Optional[torch.device] = None

def load_ml_model():
    """
    サーバー起動時に一度だけ呼び出されるモデル読み込み関数
    """
    global model, device
    if model is not None:
        print("✅ モデルはすでにロード済みです。")
        return

    print("⌛ 機械学習モデルをロード中です...")
    
    # ==========================
    # モデル構築
    # ==========================
    custom_backbone = resnet18(pretrained=False)
    out_channels = 256
    backbone_fpn = _resnet_fpn_extractor(
        custom_backbone,
        trainable_layers=5,
        extra_blocks=LastLevelP6P7(out_channels, out_channels)
    )

    base_sizes = [8, 16, 32, 64, 128, 256]
    sizes_for_anchor = tuple((s,) for s in base_sizes[:6])
    anchor_generator = AnchorGenerator(
        sizes=sizes_for_anchor,
        aspect_ratios=((0.5, 1.0, 2.0),) * 6
    )

    NUM_CLASSES = 2
    model = RetinaNet(
        backbone=backbone_fpn,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_generator
    )

    weights_path = "retinanet_epoch20.pth" # 注: このファイルも同じフォルダに必要です
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ モデルロード完了: {weights_path} (on {device})")

# ==========================
# 推論関数（バックエンド連携用）
# ==========================
def detect_face(image_path: str, threshold: float = 0.3) -> Optional[Tuple[List[float], float]]:
    """
    画像から顔を検出し、(bbox, score)またはNoneを返す
    """
    global model, device
    if model is None or device is None:
        print("❌ エラー: モデルがロードされていません。")
        return None

    model.eval()
    img = Image.open(image_path).convert("RGB")

    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()

    # 閾値でフィルタリング
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return None  # 顔なし

    # 最もスコアが高いボックスを返す
    best_idx = scores.argmax()
    bbox = boxes[best_idx].tolist()
    score = float(scores[best_idx])

    return bbox, score

# (テスト実行用の __main__ は省略)
