async function EffectSelect(effectName){
    //effectNameを元に、選択された画像と位置を取得、描画
    //画像インポート時にバックエンドから返される内容に応じて、フロントで行う処理の内容が変動
    async function requestEffect(userRequest){                //asyncは内部に非同期処理が存在することを表す
        try{
            const response = await fetch('/get_stamp_info',{ //awaitは処理が終わるまで待機をお願いする
                method: 'POST',                           //これはPOSTメソッドです
                headers: {                                //送るデータはこの形状です
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userRequest)                         //実際に送るデータです
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

    sessionStorage.setItem("effect",effectName);
    const userID = sessionStorage.getItem("ID");
    const userRequest = {                           //送る内容を封筒に収める
        upload_image_id: userID,
        stamp_id: effectName
    }
    const result = await requestEffect(userRequest);    //手紙を送って、返信を格納できるまで少し待つ
    console.log("エフェクト画像もらえたかな？");
    
    const ImageSpace = document.getElementById('ImageSpace');   //描画領域となるcanvasを指定
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const Img = new Image();
    Img.src = JSON.parse(sessionStorage.getItem("Img"));        //画像読み込み開始

    Img.onload = () => {                                        //画像読み込み終わった後の処理
        context.clearRect(0,0,ImageSpace.clientWidth,ImageSpace.clientHeight);  //一回全消し
        const scale = ImageSpace.width/Img.width;
        const effectImg = new Image();
        effectImg.src = result["base_image_url"];
        const effectX = result["x"];
        const effectY = result["y"];
        const effectScale = result["scale"];
        context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        context.drawImage(effectImg, effectX, effectY, effectImg.width*effectScale, effectImg.height*effectScale);
    }
}

//strage: ID, Img, effect