document.addEventListener("DOMContentLoaded", function() {  //画面更新時、sessionStorageを初期化する
    sessionStorage.clear();
});

//土台の画像として使用するものを変えて、描画し直す
async function ChangeUseImage(){
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

//strage: ID, OnEffect, UserImageScale, Img, AnotherImg