from PIL import Image, ImageDraw

def plot_results(image_path, raw_points):
    """
    ダミー関数：ランドマークプロットが無くても落ちないようにする。
    raw_points を描画した簡易画像を返す。
    """

    # 画像を開く
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # raw_points がある場合だけ丸を描く
    if raw_points:
        for p in raw_points:
            x, y = p
            r = 6
            draw.ellipse((x-r, y-r, x+r, y+r), fill="red")

    return img
