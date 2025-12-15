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
            console.error('送れなかったよ:',error.message);
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

    OnEffect.push(effectName);                                          //新しく有効化されるエフェクトを保存
    console.log(OnEffect);
    sessionStorage.setItem("OnEffect",JSON.stringify(OnEffect))         //エフェクトの有効化状況を再度保存
    
    const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const effectImg = new Image();
    //effectImg.src = result["stamp_image"];                     //エフェクト画像の読み込み開始
    //変更したよ
    await new Promise((resolve, reject) => {
    effectImg.onload = () => {                                  // 読み込み完了後、バックエンドの指示通りに画像を描画
    // 元画像 → キャンバスの拡大率（ImageImport.js で保存した値）
    const baseScale = UserImageScale;

    //     //if (effectName === "kiraeffect") {
    //         //const drawW = ImageSpace.width;
    //         //const drawH = ImageSpace.height;
    //         //context.drawImage(
    //             //effectImg,
    //             //0,          
    //             //0,
    //             //drawW,
    //             //drawH
    //         //);
    //         //return;         // ここで終了（下の通常処理には行かない）
    //         //}
    //     // バックエンドから来る座標は「元画像基準」なので、
    // // キャンバス上では baseScale 倍してあげる    
    //     const effectX = result["x"] * baseScale;
    //     const effectY = result["y"] * baseScale;
        
    //     const effectScale = result["scale"] * baseScale;
        
    //     context.drawImage(
    //         effectImg,
    //         effectX,
    //         effectY,
    //         effectImg.width  * effectScale,
    //         effectImg.height * effectScale
    //     );

        // 角度計算用に追加しました。続きます。（高井良）
        // バックエンドから中心点が送られる
        const centerX = result["x"] * baseScale;
        const centerY = result["y"] * baseScale;

        const effectScale = result["scale"] * baseScale;

        // エフェクト角度調整用
        const angle = result["angle"] || 0;

        // 回転のために幅と高さが必要なので計算
        const drawWidth = effectImg.width * effectScale;
        const drawHeight = effectImg.height * effectScale;

        // サーバーから指定された「画像のどこを中心にするか」の情報
        // なければ0.5（真ん中）を使う
        let rotationCenterX = result["rotation_center_x"] ?? 0.5; 
        let rotationCenterY = result["rotation_center_y"] ?? 0.5; 
        
        // 回転の中心点のオフセットを計算
        // 画像の左上からどれくらいズレた場所を軸にするか
        const rotationCenterXPX = drawWidth * rotationCenterX;
        const rotationCenterYPX = drawHeight * rotationCenterY;

        // 描画設定を保存
        context.save();

        // キャンバスの原点を計算した中心軸に移動
        context.translate(centerX, centerY);

        // 回転させる
        const radian = angle * (Math.PI / 180);
        context.rotate(radian);

        // 画像を描画
        // 原点を回転の中心点に移動させていたので、そこからオフセット分戻した位置に描画する
        context.drawImage(
            effectImg,
            -rotationCenterXPX,
            -rotationCenterYPX,
            drawWidth,
            drawHeight
        );

        // 描画設定を元に戻す
        context.restore();
        // 角度計算用ここまで（高井良）
        resolve();
    };
    effectImg.onerror = () => {
        console.error("画像の読み込みに失敗しました");
        resolve(); // loading が永遠に消えないのを防ぐ
    };

    effectImg.src = result["stamp_image"];
    });
}

// ★これを一番下に追加！
//function handleClick(effectName){
    //console.log("handleClick:", effectName);
    //EffectSelect(effectName);
//}

//UserImageScaleが加工する画像の倍率です
//strage: ID, OnEffect, UserImageScale, Img