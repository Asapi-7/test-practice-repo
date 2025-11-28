let count = 0;
const button = document.getElementById('toritukebutton');
button.addEventListener('click',function(){
    count++;
    if(count % 2 === 1){
        EffectSelect();
        console.log("画像を表示（奇数）")
    }else{
        EffectRemove();
        console.log("画像を消す（偶数）")
    }
});

document.addEventListener('DOMContentLoaded', () => {
    EffectRemove();
});