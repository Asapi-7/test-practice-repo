function ImageImport(files){
    if (files.length === 0) return;
    const reader = new FileReader();
    if (files.type.match("image.*")) {
		reader.onload = (event) => { 
            //web APIにデータを送って、返り値を受け取るようにする


            //描画箇所に保存した画像を描画する
            const ImageSpace = document.getElementById('ImageSpace');
            const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
            const Img = new Image();
            Img.src = "${event.target.result}";
            sessionStorage.setItem("Img", JSON.stringify(Img));         //画像を他の関数でも使えるよう保存しておく
            context.drawImage(Img, 50, 50, 500, 400);
        }
        reader.readAsDataURL(files);                                     //これに成功するとreader.onloadが動き出す
	}
}