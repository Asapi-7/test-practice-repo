async function EffectSelect(effectName){
    //effectNameを元に、選択された画像と位置を取得、描画
    async function requestEffect(userRequest){               //asyncは内部に非同期処理が存在することを表す
        try{
            const response = await fetch('/get_stamp_info',{ //awaitは処理が終わるまで待機をお願いする
                method: 'POST',                             //これはPOSTメソッドです
                headers: {                                  //送るデータはこの形状です
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userRequest)           //実際に送るデータです
            });
            const result = await response.json();
            return result;
        }catch(error){
            console.error('送れなかったよ:',error);
            return null
            }
        }

    const userID = sessionStorage.getItem("ID");                        //保存していたIDを回収する
    const OnEffect = JSON.parse(sessionStorage.getItem("OnEffect"));    //保存していたエフェクトの有効化状況を回収する
    const UserImageScale = JSON.parse(sessionStorage.getItem("UserImageScale"));    //ユーザーが入れた画像の表示倍率を回収する
    console.log(userID);
    
    if(userID === null){  //まだ加工する画像を選択していなかった場合 ===だと型も同じじゃないとtrueにならない(0=="0"T,0==="0"F)
        console.log("エラーってるよ！画像入れてないから！");
        alert("先に加工する画像を選択してください。");
        return;
    }

    OnEffect.push(effectName);                                          //新しく有効化されるエフェクトを保存
    console.log(OnEffect);
    sessionStorage.setItem("OnEffect",JSON.stringify(OnEffect))         //エフェクトの有効化状況を再度保存
    const userRequest = {                                               //送る内容を封筒に収める
        upload_image_id: userID,
        stamp_id: effectName
    }
    const result = await requestEffect(userRequest);    //手紙を送って、返信を格納できるまで少し待つ

    if(result.detail){
        console.log("エラーってるよ!backで!");
        alert(result.detail);
        return;
    }
    
    const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const effectImg = new Image();
    effectImg.src = result["stamp_image"];                     //エフェクト画像の読み込み開始
    //変更したよ
    effectImg.onload = () => {
        
        const baseScale = UserImageScale ?? 1;
        
        if (effectName === "kiraeffect") {
            const drawW = ImageSpace.width;
            const drawH = ImageSpace.height;
            context.drawImage(
                effectImg,
                0,          
                0,
                drawW,
                drawH
            );
            return;         // ここで終了（下の通常処理には行かない）
            }
        // バックエンドから来る座標は「元画像基準」なので、
    // キャンバス上では baseScale 倍してあげる    
        const effectX = result["x"] * baseScale;
        const effectY = result["y"] * baseScale;
        
        const effectScale = result["scale"] * baseScale;
        
        context.drawImage(
            effectImg,
            effectX,
            effectY,
            effectImg.width  * effectScale,
            effectImg.height * effectScale
        );
    };
    effectImg.src = result["stamp_image"];

}

// ★これを一番下に追加！
function handleClick(effectName){
    console.log("handleClick:", effectName);
    EffectSelect(effectName);
}

//UserImageScaleが加工する画像の倍率です
//strage: ID, OnEffect, UserImageScale, Img