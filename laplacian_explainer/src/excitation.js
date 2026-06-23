"use strict";
//=============================================================================
// СЕКЦИЯ «Краевое возбуждение e → формозависимые амплитуды (shape)» (REQ 2).
// Самодостаточна: собственная форма/мода/shape, НЕ трогает песочницу (modeMask,
// computeLaplacian). Переиспользует jacobiEigen, NEIGH, $ из ранее загруженных
// модулей. Проекция e→φ считается бит-в-бит формулой casynth_core.map_laplacian.
//=============================================================================

const EX_F0 = 220;          // фикс. нота секции (проекция/произведения от f0 не зависят)

// Пресеты как строки "1"/"0" по рядам (tight bbox каждой формы).
const EX_PRESETS = {
  string:   ["11111111"],                 // «струна» 1×8 — герой B+C (моды = гармоники)
  block:    ["1111","1111","1111"],        // компактный блок 3×4 = 12 клеток
  snake:    ["11111","00001","11111","10000"], // протяжённая змейка = 12 клеток
  blinker:  ["111"],                       // 1×3 — симметрия глушит антисимм. моду
  blinkerp: ["111","001"],                 // blinker+1 (сломана симметрия) — мода оживает
};
function exParse(rows){ return rows.map(s => s.split("").map(ch => ch === "1" ? 1 : 0)); }

//=================== СОСТОЯНИЕ СЕКЦИИ ===================
let exForm   = exParse(EX_PRESETS.string);
let exSel    = 0;       // выбранная мода (rank среди звучащих)
let exShape  = 0;       // слайдер shape для панели C
let exDef    = false;   // панель A: deg (false) ↔ дефицит 8−deg (true)
let exData   = null;    // результат exBuild()

//=================== ПОСТРОЕНИЕ (граф → L → λ,φ → e → моды) ===================
function exBuild(){
  const form = exForm, rows = form.length, cols = form[0].length;
  const live = [];
  for(let r=0;r<rows;r++) for(let c=0;c<cols;c++) if(form[r][c]) live.push([r,c]);
  const cnt = live.length;
  const out = {rows,cols,live,cnt,L:null,eigs:[],vectors:[],e:[],modes:[],comp:0,note:""};
  if(cnt < 2){ out.note = "меньше 2 клеток → тишина"; return out; }

  const key=(r,c)=>r+","+c;
  const pos=new Map(); live.forEach(([r,c],i)=>pos.set(key(r,c),i));
  const L=Array.from({length:cnt},()=>Array(cnt).fill(0));
  for(let i=0;i<cnt;i++){ const [r,c]=live[i]; let deg=0;
    for(const [dr,dc] of NEIGH){ const j=pos.get(key(r+dr,c+dc));
      if(j!==undefined){ L[i][j]=-1; deg++; } }
    L[i][i]=deg; }
  out.L=L;
  out.e = live.map((_,i)=>L[i][i]);            // краевое возбуждение e_i = deg_i = diag(L)

  const {values, vectors} = jacobiEigen(L);
  out.eigs=values; out.vectors=vectors;
  out.comp = values.filter(v=>v<=1e-6).length; // нулевые λ = компоненты связности

  const nzOffset = values.findIndex(v=>v>1e-6);
  if(nzOffset < 0){ out.note="все λ≈0 → тишина"; return out; }
  const nz = values.slice(nzOffset).map(v=>Math.sqrt(Math.max(v,0)));
  const scale = EX_F0 / nz[0];                 // низшая ненулевая → f0 (как sel[0] в ядре)

  // звучащие моды (все ненулевые ниже guard); rank = порядок, маска = все включены
  const modes=[];
  for(let i=0;i<nz.length;i++){
    const freq = nz[i]*scale;
    if(freq >= GUARD) continue;
    const eigIdx = nzOffset + i;               // выравнивание φ по той же моде, что дала частоту
    const phi = vectors[eigIdx];
    let dot=0; for(let j=0;j<cnt;j++) dot += out.e[j]*phi[j];
    modes.push({rank:modes.length, eigIdx, freq, dot, proj:Math.abs(dot), phi});
  }
  // амплитуды — паритет с ядром (α=1, маска=все): rolloffN[0]=1, projN=proj/max(proj)
  const projMx = Math.max(1e-9, ...modes.map(m=>m.proj));
  modes.forEach((m,i)=>{ m.rolloffN = 1/(i+1); m.projN = m.proj/projMx; });
  out.modes=modes; out.f0=EX_F0;
  return out;
}
// нормированный итоговый столбик при заданном shape (blend → нормировка max=1)
function exAmpN(modes, s){
  const blend = modes.map(m => (1-s)*m.rolloffN + s*m.projN);
  const mx = Math.max(1e-9, ...blend);
  return blend.map(b => b/mx);
}

//=================== ЦВЕТА ===================
function exLerp(a,b,t){ return a + (b-a)*t; }
function exHeat(v, vmax){                       // тепло: тёмный → оранж → красный
  const t = vmax>1e-9 ? Math.max(0,Math.min(1,v/vmax)) : 0;
  let r,g,b;
  if(t<0.5){ const u=t/0.5; r=exLerp(30,255,u); g=exLerp(34,150,u); b=exLerp(42,40,u); }
  else{ const u=(t-0.5)/0.5; r=255; g=exLerp(150,60,u); b=exLerp(40,50,u); }
  return `rgb(${r|0},${g|0},${b|0})`;
}
function exPhi(v, vmax){                         // фигура Хладни: + красный / − синий / 0 белый
  const t = vmax>1e-9 ? v/vmax : 0;
  const R = t>0?255:Math.round(255*(1+t));
  const B = t<0?255:Math.round(255*(1-t));
  const G = Math.round(255*(1-Math.abs(t)));
  return `rgb(${R},${G},${B})`;
}
function exProd(v, vmax){                         // вклад > 0 зелёный / гасит < 0 малиновый
  const t = vmax>1e-9 ? Math.max(-1,Math.min(1,v/vmax)) : 0;
  if(t>=0){ const u=t;  return `rgb(${exLerp(40,60,u)|0},${exLerp(44,230,u)|0},${exLerp(54,90,u)|0})`; }
  const u=-t;          return `rgb(${exLerp(40,235,u)|0},${exLerp(44,70,u)|0},${exLerp(54,120,u)|0})`;
}

//=================== ОТРИСОВКА ФОРМЫ (общая) ===================
function exDrawForm(ctx, W, H, b, vals, colorFn, vmax){
  ctx.clearRect(0,0,W,H);
  if(!b || b.cnt<1){ ctx.fillStyle="#9aa0a6"; ctx.font="13px monospace"; ctx.fillText("пусто",10,20); return; }
  const cell = Math.max(8, Math.floor(Math.min((W-16)/b.cols, (H-16)/b.rows)));
  const ox = (W-cell*b.cols)/2, oy = (H-cell*b.rows)/2;
  const valOf=new Map(); b.live.forEach(([r,c],i)=>valOf.set(r+","+c, vals?vals[i]:0));
  for(let r=0;r<b.rows;r++) for(let c=0;c<b.cols;c++){
    const x=ox+c*cell, y=oy+r*cell;
    ctx.strokeStyle="#1c2129"; ctx.strokeRect(x,y,cell,cell);
    const k=r+","+c;
    if(valOf.has(k)){ ctx.fillStyle=colorFn(valOf.get(k),vmax); ctx.fillRect(x+1,y+1,cell-2,cell-2); }
  }
}
function exMaxAbs(arr){ return Math.max(1e-9, ...arr.map(Math.abs)); }

//=================== ПАНЕЛЬ A — форма звучит границей ===================
const exAc = $("exA").getContext("2d");
function exDrawA(){
  const b=exData; if(!b){ return; }
  const vals = exDef ? b.e.map(v=>8-v) : b.e.slice();
  const vmax = Math.max(1, ...vals);
  exDrawForm(exAc, 300, 200, b, vals, exHeat, vmax);
  $("exAcap").innerHTML = exDef
    ? "режим <b>дефицит 8−deg</b>: однородная внутренность (deg=8) гаснет в 0 — светится только кромка."
    : "режим <b>deg</b>: цвет клетки = число живых 8-соседей (диагональ L). Внутренность ярче, край темнее.";
}

//=================== ПАНЕЛЬ B — у моды есть форма (Хладни) + solo ===================
const exBc = $("exB").getContext("2d");
function exDrawB(){
  const b=exData; if(!b) return;
  if(exSel<0 || !b.modes[exSel]){ exDrawForm(exBc,300,200,b,null,exHeat,1);
    $("exBcap").textContent="выбери моду в полоске ниже."; return; }
  const phi=b.modes[exSel].phi;
  exDrawForm(exBc, 300, 200, b, phi, exPhi, exMaxAbs(phi));
  $("exBcap").innerHTML = `мода <b>m${exSel+1}</b> · частота <b>${exData.modes[exSel].freq.toFixed(1)} Гц</b>. `
    + `Красный=+, синий=−, белый=узел (≈0). На «струне» это гармоники: ниже — плавный градиент, выше — частые чередования.`;
}

//=================== ПОЛОСКА ВЫБОРА МОДЫ ===================
const exEigCtx = $("exEig").getContext("2d");
let exEigBars=[];
function exDrawEig(){
  const W=640,H=92; exEigCtx.clearRect(0,0,W,H); exEigBars=[];
  const m = exData ? exData.modes : [];
  if(!m.length){ exEigCtx.fillStyle="#9aa0a6"; exEigCtx.font="13px monospace";
    exEigCtx.fillText("нет звучащих мод",10,20); return; }
  const fmax=Math.max(...m.map(x=>x.freq)), pad=12, bw=(W-2*pad)/m.length, base=H-22;
  for(let i=0;i<m.length;i++){
    const h=(m[i].freq/fmax)*(base-10), x=pad+i*bw+2, y=base-h, w=Math.max(4,bw-6);
    exEigCtx.fillStyle = (i===exSel)?"#69db7c":"#ffe066";
    exEigCtx.fillRect(x,y,w,h);
    exEigBars.push({x,y,w,base,i});
    exEigCtx.fillStyle="#9aa0a6"; exEigCtx.font="10px monospace";
    exEigCtx.fillText("m"+(i+1), x, H-6);
  }
}
$("exEig").addEventListener("click",e=>{
  const rect=$("exEig").getBoundingClientRect();
  const x=(e.clientX-rect.left)*(640/rect.width);
  for(const b of exEigBars){ if(x>=b.x-2 && x<=b.x+b.w+2){ exSel=b.i; exRedraw(); return; } }
});

//=================== ПАНЕЛЬ C — громкость = перекрытие удара с модой ===================
const exCe   = $("exCe").getContext("2d");
const exCphi = $("exCphi").getContext("2d");
const exCpr  = $("exCprod").getContext("2d");
const exCbar = $("exCbars").getContext("2d");
function exDrawC(){
  const b=exData; if(!b || exSel<0 || !b.modes[exSel]){ return; }
  const phi=b.modes[exSel].phi;
  const prod = b.e.map((ei,j)=>ei*phi[j]);
  exDrawForm(exCe,  150,130, b, b.e, exHeat, Math.max(1,...b.e));
  exDrawForm(exCphi,150,130, b, phi, exPhi, exMaxAbs(phi));
  exDrawForm(exCpr, 150,130, b, prod, exProd, exMaxAbs(prod));

  // три столбика: rolloff (shape=0) · проекция (shape=1) · итог (текущий shape)
  const m=b.modes[exSel];
  const ampN = exAmpN(b.modes, exShape)[exSel];
  const bars=[["1/iᵅ",m.rolloffN,"#5a6172"],["проекция",m.projN,"#74c0fc"],["итог",ampN,"#ffe066"]];
  const W=220,H=130; exCbar.clearRect(0,0,W,H);
  const base=H-22, bw=46, gap=20, x0=24;
  exCbar.strokeStyle="#2b313c"; exCbar.beginPath(); exCbar.moveTo(10,base); exCbar.lineTo(W-6,base); exCbar.stroke();
  bars.forEach(([lbl,val,col],i)=>{
    const x=x0+i*(bw+gap), h=Math.max(1,val*(base-12)), y=base-h;
    exCbar.fillStyle=col; exCbar.fillRect(x,y,bw,h);
    exCbar.fillStyle="#9aa0a6"; exCbar.font="11px monospace"; exCbar.textAlign="center";
    exCbar.fillText(lbl, x+bw/2, H-6);
    exCbar.fillStyle="#e6e9ef"; exCbar.fillText(val.toFixed(2), x+bw/2, y-4);
  });
  exCbar.textAlign="left";

  const dot=m.dot;
  $("exCcap").innerHTML =
    `Σ <span class="mono">e·φ</span> = <b class="${Math.abs(dot)<1e-3?'bad':'ok'}">${dot.toFixed(3)}</b> `
    + `→ |·| = <b>${m.proj.toFixed(3)}</b> → нормир. проекция <b>${m.projN.toFixed(3)}</b>. `
    + `Поклеточно: зелёный — удар и мода совпали по знаку (вклад), малиновый — противоположны (гасят).`;
}

//=================== ПАНЕЛЬ D — почему симметрия глушит моду ===================
function exDrawD(){
  const b=exData; if(!b){ $("exDcap").textContent=""; return; }
  if(exSel<0 || !b.modes[exSel]){ $("exDcap").textContent="нет выбранной моды."; return; }
  const m=b.modes[exSel];
  const silenced = Math.abs(m.dot) < 1e-3;
  const ampN = exAmpN(b.modes, 1)[exSel];   // итог при shape=1 (чистая проекция)
  let h = `Текущая мода <b>m${exSel+1}</b>: ненормир. сумма <span class="mono">⟨e,φ⟩</span> = `
        + `<b class="${silenced?'bad':'ok'}">${m.dot.toFixed(3)}</b>, нормированный столбик при shape=1 = `
        + `<b>${ampN.toFixed(3)}</b>. `;
  if(silenced){
    h += `<span class="bad">Мода заглушена симметрией</span>: удар <span class="mono">e</span> симметричен, `
       + `мода антисимметрична → произведение <span class="mono">+0−</span> суммируется в 0. `
       + `Загрузи <b>Blinker+1</b> — симметрия сломана, та же мода оживает.`;
  } else {
    h += `Симметрия нарушена (или мода симметрична удару) → ⟨e,φ⟩≠0, мода звучит. `
       + `Сравни с <b>Blinker</b> (m1): там ⟨e,φ⟩=0 → столбик ноль. `
       + `«Симметричная пластина не звенит антисимметричной модой».`;
  }
  $("exDcap").innerHTML = h;
}

//=================== ПЕРЕСЧЁТ ===================
function exRedraw(){ exDrawA(); exDrawB(); exDrawEig(); exDrawC(); exDrawD(); }
function exRecompute(){
  exData = exBuild();
  const M = exData.modes.length;
  if(M===0) exSel=-1;
  else if(exSel<0 || exSel>=M) exSel=0;
  exRedraw();
}

//=================== SOLO-АУДИО (свой контекст) ===================
let exActx=null, exOsc=null, exGain=null;
function exPlaySolo(freq){
  if(!exActx){
    exActx=new (window.AudioContext||window.webkitAudioContext)();
    exOsc=exActx.createOscillator(); exOsc.type="sine";
    exGain=exActx.createGain(); exGain.gain.value=0.0001;
    exOsc.connect(exGain); exGain.connect(exActx.destination); exOsc.start();
  }
  if(exActx.state==="suspended") exActx.resume();
  const t=exActx.currentTime;
  exOsc.frequency.setValueAtTime(freq,t);
  exGain.gain.cancelScheduledValues(t);
  exGain.gain.setValueAtTime(0.0001,t);
  exGain.gain.exponentialRampToValueAtTime(0.25,t+0.02);
  exGain.gain.exponentialRampToValueAtTime(0.0001,t+0.9);
}

//=================== СОБЫТИЯ ===================
document.querySelectorAll(".exp").forEach(btn=>{
  btn.addEventListener("click",()=>{
    const p=EX_PRESETS[btn.dataset.form]; if(!p) return;
    exForm=exParse(p);
    exSel=0;                       // m1 — низшая (для blinker = антисимм., демо панели D)
    document.querySelectorAll(".exp").forEach(x=>x.classList.remove("on"));
    btn.classList.add("on");
    exRecompute();
  });
});
$("exDef").addEventListener("click",()=>{ exDef=!exDef; $("exDef").classList.toggle("on",exDef); exDrawA(); });
$("exShape").addEventListener("input",()=>{
  exShape=+$("exShape").value; $("exShapev").textContent=exShape.toFixed(2); exDrawC();
});
$("exSolo").addEventListener("click",()=>{
  if(exData && exSel>=0 && exData.modes[exSel]) exPlaySolo(exData.modes[exSel].freq);
});

//=================== СТАРТ ===================
(function(){ const sb=document.querySelector('.exp[data-form="string"]'); if(sb) sb.classList.add("on"); })();
exRecompute();
