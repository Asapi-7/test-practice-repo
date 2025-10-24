function ImageImport(files){
    if (files.length === 0) return;
		const file = files[0];              //もらうデータは必ずファイル群になってるから、先頭だけ抜き出して画像のみにする
    const reader = new FileReader();
    if (file.type.match("image.*")) {
		reader.onload = (event) => { 
            //web APIにデータを送って、返り値を受け取るようにする
            async function sendUserImage(userImage){                //asyncは内部に非同期処理が存在することを表す
              try{
                const response = await fetch('/upload_and_detect',{ //awaitは処理が終わるまで待機をお願いする
                  method: 'POST',                           //これはPOSTメソッドです
                  headers: {                                //送るデータはこの形状です
                    'Content-Type': 'multipart/form-data'
                  },
                  file: userImage                           //実際に送るデータです
                });
                if(!response.ok){
                  throw new Error('返答が芳しくなかった');
                }
                const result = await response.json();
                return result;
              }catch(error){
                console.error('送れなかったよ:',error.message);
                return null
              }
            }

            const UserID = sendUserImage(file);
            sessionStorage.setItem("ID", UserID);
            console.log(UserID);


            //描画箇所に保存した画像を描画する
            const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
            const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
            const Img = new Image();
            Img.src = event.target.result;                              //画像読み込み開始

            Img.onload = () => {
              const scale = ImageSpace.width/Img.width;
              context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
              console.log("deteru?");
              sessionStorage.setItem("Img", JSON.stringify(event.target.result));         //画像を他の関数でも使えるよう保存しておく    
            }
        }
        reader.readAsDataURL(file);                                     //これに成功するとreader.onloadが動き出す
	}
}