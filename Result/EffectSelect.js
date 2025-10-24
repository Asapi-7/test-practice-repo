function EffectSelect(effectName){
    //effectNameを元に、選択された画像と位置を取得、描画
    //画像インポート時にバックエンドから返される内容に応じて、フロントで行う処理の内容が変動
    
    const ImageSpace = document.getElementById('ImageSpace');
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const Img = new Image();
    Img.src = JSON.parse(sessionStorage.getItem("Img"));

    Img,Onload = () => {
        context.clearRect(0,0,ImageSpace.clientWidth,ImageSpace.clientHeight);
        const scale = canvas.width/Img.width;
        context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        //context.drawImage(Img, 50, 50, 500, 400);エフェクトこれ
        sessionStorage.setItem("select", JSON.stringify(effectName));
    }
}