# mizunuma_model.py - 水沼さん/唯ちゃんのモデル

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# OpenMPライブラリの重複読み込み警告を抑制
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cpu")
mizunuma_model = None

# =================================================================
# モデルクラス定義
# =================================================================

class BasicBlock(nn.Module):
    '''
    ResNet18における残差ブロック
    in_channels : 入力チャネル数
    out_channels: 出力チャネル数
    stride      : 畳み込み層のストライド
    '''
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()

        # 残差接続
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # スキップ接続のダウンサンプリング (寸法合わせ)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor):
        identity = x  # 恒等写像 (スキップ接続) を保存

        # 残差写像
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # ダウンサンプリング処理
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 残差写像と恒等写像の要素毎の和を計算
        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    '''
    ResNet18モデル
    num_classes: 分類対象の物体クラス数 (ランドマーク回帰用に置き換えられる)
    '''
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, return_embed: bool=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.flatten(1)

        if return_embed:
            return x

        x = self.linear(x)
        return x


class LandmarkRegressor(nn.Module):
    '''
    ランドマーク回帰モデル (9点 = 18座標)
    '''
    def __init__(self, num_landmarks=9):
        super(LandmarkRegressor, self).__init__()
        
        # 1. Backbone: カスタムResNet18を使用
        self.backbone = ResNet18(num_classes=1000)
        
        # 2. Head: Dense層 (最終層) の変更
        num_features = self.backbone.linear.in_features
        
        # 3. 出力層をランドマークの数 (18) に置き換え
        self.backbone.linear = nn.Linear(num_features, num_landmarks * 2)

    def forward(self, x):
        return self.backbone(x)


# =================================================================
# モデルのロードと推論関数
# =================================================================

def load_ml_model():
    """ゆいちゃんのモデルをロード"""
    global mizunuma_model
    try:
        weight_path = os.path.join(os.path.dirname(__file__), "landmark_regressor_final.pth")
        print(f"モデルローディング中: {weight_path}")
        
        mizunuma_model = LandmarkRegressor(num_landmarks=9)
        mizunuma_model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        mizunuma_model.to(DEVICE)
        mizunuma_model.eval()
        
        print("モデルのロード完了")
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        mizunuma_model = None


def detect_face_and_lndmk(image_path: str, score_threshold: float = 0.3):
    """
    ゆいちゃんのモデルで顔のランドマークを検出
    
    Args:
        image_path: 画像ファイルのパス
        score_threshold: 未使用（他のモデルとのインターフェース統一用）
    
    Returns:
        list: [[x_min, y_min], [x_max, y_max], [x1, y1], [x2, y2], ..., [x9, y9]]
              最初の2点はバウンディングボックス、その後9点がランドマーク座標
              エラー時はNoneを返す
    """
    if mizunuma_model is None:
        print("❌ モデルが読み込まれていません")
        return None
    
    start_time = time.time()
    
    try:
        # 画像の読み込み
        img = Image.open(image_path).convert("RGB")
        original_w, original_h = img.size
        
        # モデルの入力サイズ
        IMG_SIZE = 224
        
        # 前処理（ImageNetの統計値で正規化）
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 推論
        with torch.no_grad():
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)
            output = mizunuma_model(input_tensor).cpu().numpy()[0]
        
        # ランドマーク座標を (9, 2) に変換
        landmarks = output.reshape(9, 2)
        
        # 元の画像サイズにスケール変換
        landmarks[:, 0] *= (original_w / IMG_SIZE)
        landmarks[:, 1] *= (original_h / IMG_SIZE)
        
        # バウンディングボックスを計算（ランドマークから推定）
        x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
        y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
        
        # マージンを追加（顔全体を含むように20%拡大）
        margin_x = (x_max - x_min) * 0.2
        margin_y = (y_max - y_min) * 0.2
        
        # 結果のフォーマット: [bbox_min, bbox_max, landmark1, ..., landmark9]
        result = [
            [float(max(0, x_min - margin_x)), float(max(0, y_min - margin_y))],  # bbox左上
            [float(min(original_w, x_max + margin_x)), float(min(original_h, y_max + margin_y))]  # bbox右下
        ]
        
        # ランドマーク座標を追加
        result.extend([[float(x), float(y)] for x, y in landmarks])
        
        # 信頼度スコアを計算（ランドマークが画像内に収まっているかで判定）
        # 全てのランドマークが画像内にある場合は高スコア
        valid_landmarks = sum([
            1 for x, y in landmarks 
            if 0 <= x <= original_w and 0 <= y <= original_h
        ])
        confidence_score = valid_landmarks / 9.0  # 9点中何点が有効か
        
        elapsed = time.time() - start_time
        print(f"モデル検出完了 (信頼度: {confidence_score:.2f}, 実行時間: {elapsed:.3f}秒)")
        
        # 戻り値: (result, confidence_score)
        return result, float(confidence_score)
        
    except Exception as e:
        print(f"モデル検出エラー: {e}")
        return None
