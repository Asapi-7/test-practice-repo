function ImageDownload(){
    //画像をダウンロードさせる
    const ImageSpace = document.getElementById('ImageSpace');
	const blob = ImageSpace.toBlob();   							 //バイナリファイルを扱う
	const link = document.createElement("a");               	     //リンクになるタグを作る
	link.href = URL.createObjectURL(blob);               	         //中に入れるリンクを指定(生成したテキストファイル)
	link.download = "new_" + Img;                                    //クリックするとダウンロードするよって教えてる
	link.click();
}