document.addEventListener("DOMContentLoaded", function() {  //画面更新時、sessionStorageを初期化する
    sessionStorage.clear();
});

//土台の画像として使用するものを変えて、描画し直す
async function ChangeUseImage(){
    const userID = sessionStorage.getItem("ID");                        //保存していたIDを回収する
    if(userID === null){  //まだ加工する画像を選択していなかった場合 ===だと型も同じじゃないとtrueにならない(0=="0"T,0==="0"F)
        console.log("エラーってるよ！画像入れてないから！");
        alert("先に加工する画像を選択してください。");
        return;
    }

    const check = await CheckSession();
    if(!check){        //処理の中でFastAPIサーバーとの通信を行うので、事前に接続が可能かを確認する
      console.log("エラーってるよ!backで!Check動いてるよ!",check);
      alert("長時間操作が無かったため、接続が切れました。再度画像を選択してください。");
      return;
    }

    const RegenerateEffect = JSON.parse(sessionStorage.getItem("OnEffect"));    //現時点で描画されているエフェクトを得る
    sessionStorage.setItem("OnEffect",JSON.stringify([]));                      //エフェクトを再度描画し直すため、一回有効化状況を空っぽに
    const keep = sessionStorage.getItem("Img");
    sessionStorage.setItem("Img", sessionStorage.getItem("AnotherImg"));
    sessionStorage.setItem("AnotherImg", keep);

    const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const Img = new Image();
    Img.src = JSON.parse(sessionStorage.getItem("Img"));        //画像読み込み開始

    Img.onload = async () => {                                        //画像読み込み終わった後の処理
        context.clearRect(0,0,ImageSpace.clientWidth,ImageSpace.clientHeight);  //一回全消し
        if(Img.width <= Img.height){                                            //元画像描画し直し
          const scale = ImageSpace.height/Img.height;
          ImageSpace.setAttribute('width', Img.width*scale)
          context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        }else{
          const scale = ImageSpace.width/Img.width;
          ImageSpace.setAttribute('height', Img.height*scale)
          context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        }                                                                       //ここまでで元画像描画完了
        for(let i=0; i<RegenerateEffect.length; i++){                         //再度エフェクトを描画し直す
            await EffectSelect(RegenerateEffect[i]);
        }
    }
}

//接続状況をFastAPIに確認し、結果を返す関数
async function CheckSession(){
  async function requestSessionNow(userRequest){               //asyncは内部に非同期処理が存在することを表す
        try{
            const response = await fetch('/check_ID',{      //awaitは処理が終わるまで待機をお願いする
                method: 'POST',                             //これはPOSTメソッドです
                headers: {                                  //送るデータはこの形状です
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userRequest)           //実際に送るデータです
            });
            const result = await response.json();
            return result;
        }catch(error){
            console.error('送れなかったよ:',error.message);
            return null
            }
        }

    const userID = sessionStorage.getItem("ID");                        //保存していたIDを回収する
    const userRequest = {                                               //送る内容を封筒に収める
        upload_image_id: userID
    }
    const result = await requestSessionNow(userRequest);    //手紙を送って、返信を格納できるまで少し待つ

    if(result.detail){
        return false;
    }

    return JSON.stringify(result.result);
}

//strage: ID, OnEffect, UserImageScale, Img, AnotherImg