let activesubBox = null;

function opensub(name){
    const newsubBox = document.getElementById("sub_" + name);
    if (activesubBox && activesubBox !== newsubBox) {
        activesubBox.style.display = "none";
    }

    if (activesubBox === newsubBox){
        newsubBox.style.display = "none";
        activesubBox = null;
        return;
    }

    newsubBox.style.display = "flex";
    activesubBox = newsubBox;
}
//hideallsubs()は一旦コメントで残す。
//function hideallsubs() {
    //const all = document.getElementsByClassName("subButtons");
    //for (let sub of all) {
        //sub.style.display = "none";
    //}
//}

function applyeffect(effectName){
    console.log("エフェクト：",effectName);
    handleClick(effectName);
    //activesubBox = null;
}