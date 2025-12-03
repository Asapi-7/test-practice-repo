// ボタンごとに押した回数を保持
const pressCount = {};

function handleClick(effectName) {
    // 初回クリック時は 0 → 1 にする
    if (!pressCount[effectName]) pressCount[effectName] = 0;

    // カウントを増やす
    pressCount[effectName]++;

    // 奇数回／偶数回を判定して関数呼び出し
    if (pressCount[effectName] % 2 === 1) {
        EffectSelect(effectName);   // 奇数 → 選択
    } else {
        EffectRemove(effectName);   // 偶数 → 削除
    }
}
