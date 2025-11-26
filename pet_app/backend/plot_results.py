import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

LANDMARK_COLOR = 'red'
BBOX_COLOR = 'green'
POINT_SIZE = 10
LINE_WIDTH = 3

# image_path (str): 描画対象の画像ファイルへのパス
# results: 予測結果データ ( detect_face_and_lndmk関数が返す値 )
# output_path (str): 描画結果の画像を保存するファイルパス
def plot_results_and_save(image_path: str, results, output_path: str):
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return

    #BBoxとランドマーク座標の取得
    bbox_min = results[0]
    bbox_max = results[1]
    landmarks = np.array(results[2:]) # 9点のランドマーク座標

    #Matplotlibを使ってプロット
    plt.figure(figsize=(12, 12))
    plt.imshow(img_pil)
    
    # BBoxのプロット
    xmin, ymin = bbox_min
    xmax, ymax = bbox_max
    
    # BBoxを描画
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, edgecolor=BBOX_COLOR, linewidth=LINE_WIDTH)
    plt.gca().add_patch(rect)
    
    # ランドマークのプロット (赤い点)
    # X座標 (landmarks[:, 0]) と Y座標 (landmarks[:, 1])
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c=LANDMARK_COLOR, s=POINT_SIZE, zorder=10)
    
    plt.axis('off') 
    
    # 4. 画像を保存
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()