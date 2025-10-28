function ImageImport(files){
  if (files.length === 0) return;
	const file = files[0];              //もらうデータは必ずファイル群になってるから、先頭だけ抜き出して画像のみにする
  const reader = new FileReader();
  if (file.type.match("image.*")) {
	  reader.onload = async (event) => { 
      //web APIにデータを送って、返り値を受け取るようにする
      async function sendUserImage(file){            //asyncは内部に非同期処理が存在することを表す
        const formData = new FormData();
        formData.append('file', file);                        //送る内容を封筒に収める 
        try{
          const response = await fetch('/upload_and_detect',{ //awaitは処理が終わるまで待機をお願いする
            method: 'POST',                                   //これはPOSTメソッドです
            body: formData                                    //実際に送るデータです
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
    
      const result = await sendUserImage(file);   //手紙を送って、返信を格納できるまで少し待つ
      const userID = result["upload_image_id"];
      sessionStorage.setItem("ID", userID);
      console.log(userID);

      //描画箇所に保存した画像を描画する
      const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
      const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
      const Img = new Image();
      Img.src = event.target.result;                              //画像読み込み開始

      Img.onload = () => {                                        //画像読み込み終わった後の処理
        const scale = ImageSpace.width/Img.width;
        context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        console.log("deteru?");
        sessionStorage.setItem("Img", JSON.stringify(event.target.result));         //画像を他の関数でも使えるよう保存しておく    
      }
    }
    reader.readAsDataURL(file);                                     //これに成功するとreader.onloadが動き出す
	}
}

//strage: ID, Img