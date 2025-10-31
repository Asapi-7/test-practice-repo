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
      const userID = result["upload_image_id"];   //返答からIDの情報を取る
      sessionStorage.setItem("ID", userID);       //ユーザー自身でIDを保持
      console.log(userID);

      //描画箇所に保存した画像を描画する
      const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
      const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得　指定したcanvas専用のお絵描き道具得る感じ？
      ImageSpace.setAttribute('width', '650');                    //画像再インポート時、canvasサイズを元の大きさに戻したり一回全消ししたり
      ImageSpace.setAttribute('height', '650');
      context.clearRect(0,0,ImageSpace.clientWidth,ImageSpace.clientHeight);
      const Img = new Image();                                    //ここに画像が入る
      Img.src = event.target.result;                              //画像読み込み開始　こいつは処理が長い

      Img.onload = () => {                                        //画像読み込み終わった後の処理
        if(Img.width <= Img.height){                              //画像が縦長か横長かによって、幅を合わせる方を変更
          const scale = ImageSpace.height/Img.height;
          ImageSpace.setAttribute('width', Img.width*scale)
          context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        }else{
          const scale = ImageSpace.width/Img.width;
          ImageSpace.setAttribute('height', Img.height*scale)
          context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        }
        console.log("deteru?");
        sessionStorage.setItem("Img", JSON.stringify(event.target.result));         //画像を他の関数でも使えるよう保存しておく    
      }
    }
    reader.readAsDataURL(file);                                     //画像をURLに変換　これに成功するとreader.onloadが動き出す
	}
}

//strage: ID, Img