async function EffectRemove(effectName){
    //effectNameを元に、選択されたエフェクト画像を削除した画像を出力する
    //画像の再生成には、EffectSelectを最大限活用している
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

    let RegenerateEffect = JSON.parse(sessionStorage.getItem("OnEffect"));    //現時点で描画されているエフェクトを得る
    RegenerateEffect = RegenerateEffect.filter(item => item !== effectName);    //今回削除するエフェクトだけ、配列から消す
    sessionStorage.setItem("OnEffect",JSON.stringify([]));                      //エフェクトを再度描画し直すため、一回有効化状況を空っぽに

    const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const Img = new Image();
    Img.src = JSON.parse(sessionStorage.getItem("Img"));        //画像読み込み開始
    await new Promise((resolve) => {
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
    
        resolve();
      
      };
    });
}

//strage: ID, OnEffect, UserImageScale, Img, AnotherImg