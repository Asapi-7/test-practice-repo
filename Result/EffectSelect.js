function EffectSelect(effectName){
    //effectNameを元に、選択された画像と位置を取得、描画
    //画像インポート時にバックエンドから返される内容に応じて、フロントで行う処理の内容が変動
    async function requestEffect(userRequest){                //asyncは内部に非同期処理が存在することを表す
        try{
            const response = await fetch('/get_stamp_info',{ //awaitは処理が終わるまで待機をお願いする
                method: 'POST',                           //これはPOSTメソッドです
                headers: {                                //送るデータはこの形状です
                    'Content-Type': 'application/json'
                },
                body: userRequest                         //実際に送るデータです
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

    const userID = sessionStorage.getItem("ID");
    const userRequest = {
        upload_image_id: userID,
        stamp_id: effectName
    }
    const response = requestEffect(userRequest);
    const result = json.loads(response);
    console.log("エフェクト画像もらえた");
    
    const ImageSpace = document.getElementById('ImageSpace');
    const context = ImageSpace.getContext('2d');                //2D描画用のコンテキストを取得
    const Img = new Image();
    Img.src = JSON.parse(sessionStorage.getItem("Img"));

    Img.onload = () => {
        context.clearRect(0,0,ImageSpace.clientWidth,ImageSpace.clientHeight);
        const scale = canvas.width/Img.width;
        //const effectImg = response.
        const effectX = result["x"];
        const effectY = result["y"];
        const effectScale = result["scale"];
        context.drawImage(Img, 0, 0, Img.width*scale, Img.height*scale);
        //context.drawImage(effectImg, effectX, effectY, effectImg.width*effectScale, effectImg.height*effectScale);エフェクトこれ
    }
}