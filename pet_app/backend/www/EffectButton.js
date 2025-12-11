let activesubBox = null;
let lock = false;

function opensub(name){
    const newsubBox = document.getElementById("sub_" + name);
    if (lock && activesubBox !== newsubBox) return;
    if (activesubBox === newsubBox){
        newsubBox.style.display = "none";
        activesubBox = null;
        lock = false;
        return;
    }

    hideallsubs();
    newsubBox.style.display = "flex";
    activesubBox = newsubBox;
    lock = true;
}

function hideallsubs() {
    const all = document.getElementsByClassName("subButtons");
    for (let sub of all) {
        sub.style.display = "none";
    }
}

function applyeffect(effectName){
    console.log("エフェクト：",effectName);
    handleClick(effectName);
    hideallsubs();
    activesubBox = null;
    lock = false;
}