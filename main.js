// 画像を読み込んでCanvasに表示する関数
function ImageImport(files) {
  // ファイルが選ばれていなければ何もしない
  if (!files || files.length === 0) {
    alert("画像ファイルを選択してください。");
    return;
  }

  const file = files[0];

  // 画像ファイルかチェック
  if (!file.type.startsWith("image/")) {
    alert("画像ファイルを選択してください。");
    return;
  }

  // Canvasとコンテキストを取得
  const canvas = document.getElementById("ImageSpace");
  const ctx = canvas.getContext("2d");

  // Imageオブジェクトを作成
  const img = new Image();

  // 読み込み完了時に描画
  img.onload = function () {
    // ① キャンバス全体を一旦クリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ② 背景を薄ピンクで塗り直す
    ctx.fillStyle = "lightpink";  // 薄いピンク
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // ③ 画像をキャンバス内にフィットさせて中央表示
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const drawWidth = img.width * scale;
    const drawHeight = img.height * scale;
    const x = (canvas.width - drawWidth) / 2;
    const y = (canvas.height - drawHeight) / 2;

    ctx.drawImage(img, x, y, drawWidth, drawHeight);
  };

  // ファイルをURLとして読み込み
  img.src = URL.createObjectURL(file);
}
