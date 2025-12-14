function resetEffects(){
  for(let key in pressCount){
    pressCount[key] = 0;
  }
}
//バグ回避
let isBusy = false;
async function ImageImport(files){
  if (isBusy) return;
  isBusy = true;
  console.log("画像インポート中制限開始"); //バグ回避
  resetEffects();
  showLoadingText();//ローディング用
  if (files.length === 0) {
  isBusy = false;
  console.log("画像インポート中制限終了");
  hideLoadingText();
  return;
  }
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
          const result = await response.json();
          return result;
        }catch(error){
          console.error('送れなかったよ:',error.message);
          return null
        }
      }
    
      const result = await sendUserImage(file);   //手紙を送って、返信を格納できるまで少し待つ
      
      if(result.detail){
          console.log("エラーってるよ!backで!");
          alert(result.detail);
          //バグ回避
          isBusy = false;
          console.log("画像インポート中制限終了");
          hideLoadingText();
          return;
      }

      const userID = result["upload_image_id"];   //返答からIDの情報を取る
      sessionStorage.setItem("ID", userID);       //ユーザー自身でIDを保持
      const landmark_plot = new Image();
      landmark_plot.src = result["landmark_plot"];   //返答からIDの情報を取る
      landmark_plot.onload = () => {
        sessionStorage.setItem("AnotherImg", JSON.stringify(result["landmark_plot"]));       //ユーザー自身でIDを保持
      }
      console.log(userID);

      const OnEffect = [];                                        //有効化しているエフェクトを持つリスト
      sessionStorage.setItem("OnEffect",JSON.stringify(OnEffect)) //エフェクトの有効化状況を保存

      //描画箇所に保存した画像を描画する
      const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
      const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得　指定したcanvas専用のお絵描き道具得る感じ？
      ImageSpace.setAttribute('width', '650');                    //画像再インポート時、canvasサイズを元の大きさに戻したり一回全消ししたり
      ImageSpace.setAttribute('height', '650');
      context.clearRect(0,0,ImageSpace.clientWidth,ImageSpace.clientHeight);
      const Img = new Image();                                    //ここに画像が入る
      //バグ回避
      Img.onerror = () => {
        alert("画像の描画に失敗しました");
        isBusy = false;
        console.log("画像インポート中制限終了");
        hideLoadingText();
      };
      Img.src = event.target.result;                              //画像読み込み開始　こいつは処理が長い

      Img.onload = () => {                                        //画像読み込み終わった後の処理
        hideLoadingText();//ローディング用
        isBusy = false;
        console.log("画像インポート中制限終了");
        let scale = 0;
        if(Img.width <= Img.height){                              //画像が縦長か横長かによって、幅を合わせる方を変更
          scale = ImageSpace.height/Img.height;
          ImageSpace.setAttribute('width', Img.width*scale)
          context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        }else{
          scale = ImageSpace.width/Img.width;
          ImageSpace.setAttribute('height', Img.height*scale)
          context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        }
        console.log("deteru?");
        sessionStorage.setItem("UserImageScale", JSON.stringify(scale));            //ユーザーが入れた画像を、描画時どれくらいのスケールに変更しているかを保存
        sessionStorage.setItem("Img", JSON.stringify(event.target.result));         //画像を他の関数でも使えるよう保存しておく    
      }
    }
    //バグ回避
    reader.onerror = () => {
    alert("画像の読み込みに失敗しました");
    isBusy = false;
    console.log("画像インポート中制限終了");
    hideLoadingText();
};
    reader.readAsDataURL(file);                                     //画像をURLに変換　これに成功するとreader.onloadが動き出す
	}else{
    alert("画像ファイルを選んでください");
    isBusy = false;
    console.log("画像インポート中制限終了");
    hideLoadingText();
    return;
  }
}

//strage: ID, OnEffect, UserImageScale, Img, AnotherImg