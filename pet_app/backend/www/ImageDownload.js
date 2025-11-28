function ImageDownload(){
    //画像をダウンロードさせる
    const ImageSpace = document.getElementById('ImageSpace');
	ImageSpace.toBlob(function(blob) { 						//canvasからバイナリを生成 toBlob((作ったバイナリファイルを引数として受け取る関数),(それが返す画像形式の指定))
		if(blob){
			const link = document.createElement("a");       //リンクになるタグを作る
			link.href = URL.createObjectURL(blob);          //中に入れるリンクを指定(生成したテキストファイル)
			link.download = "cuty_animal.png";              //クリックするとダウンロードするよって教えてる
			link.click();
			URL.revokeObjectURL(link.href);					//URLを解放する
			console.log("ダウンロードできてる？");
		}else{
			console.error("ダウンロード失敗");
		}
	}, 'image/png')
}

//strage: ID, OnEffect, UserImageScale, Img, AnotherImg