"use strict";
//=================== ОТРИСОВКА: сетка ===================
const gc=$("grid").getContext("2d"), GPX=336, CELL=GPX/SIZE;
function drawGrid(){
  gc.clearRect(0,0,GPX,GPX);
  let vec=null, vmax=1;
  if(last && selMode>=0 && last.vectors[selMode]){
    vec=last.vectors[selMode]; vmax=Math.max(1e-9,...vec.map(Math.abs));
  }
  const posIndex=new Map();
  if(last) last.live.forEach(([r,c],i)=>posIndex.set(r+","+c,i));
  for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
    const x=c*CELL,y=r*CELL;
    gc.strokeStyle="#1c2129"; gc.strokeRect(x,y,CELL,CELL);
    if(cells[r][c]){
      if(vec){
        const i=posIndex.get(r+","+c); const val=vec?vec[i]:0; const t=val/vmax;
        const R=t>0?255:Math.round(255*(1+t)), B=t<0?255:Math.round(255*(1-t)),
              G=Math.round(255*(1-Math.abs(t)));
        gc.fillStyle=`rgb(${R},${G},${B})`;
      } else gc.fillStyle="#ffe066";
      gc.fillRect(x+1,y+1,CELL-2,CELL-2);
    }
  }
}

//=================== ОТРИСОВКА: граф ===================
const grc=$("graph").getContext("2d");
function drawGraphView(){
  const W=300,H=240; grc.clearRect(0,0,W,H);
  if(!last || last.cnt<1){ grc.fillStyle="#9aa0a6"; grc.font="13px monospace";
    grc.fillText("пусто",10,20); return; }
  const pad=24, sx=(W-2*pad)/(SIZE-1||1), sy=(H-2*pad)/(SIZE-1||1);
  const P=last.live.map(([r,c])=>[pad+c*sx, pad+r*sy]);
  const key=(r,c)=>r+","+c; const pos=new Map(); last.live.forEach(([r,c],i)=>pos.set(key(r,c),i));
  grc.strokeStyle="#3a4350"; grc.lineWidth=1.5;
  for(let i=0;i<last.live.length;i++){ const [r,c]=last.live[i];
    for(const [dr,dc] of NEIGH){ const j=pos.get(key(r+dr,c+dc));
      if(j!==undefined && j>i){ grc.beginPath(); grc.moveTo(P[i][0],P[i][1]); grc.lineTo(P[j][0],P[j][1]); grc.stroke(); } } }
  let vec=null,vmax=1;
  if(selMode>=0 && last.vectors[selMode]){ vec=last.vectors[selMode]; vmax=Math.max(1e-9,...vec.map(Math.abs)); }
  const showIdx = P.length<=30;
  for(let i=0;i<P.length;i++){
    const rad = showIdx?9:7;
    grc.beginPath(); grc.arc(P[i][0],P[i][1],rad,0,7);
    if(vec){ const t=vec[i]/vmax; const R=t>0?255:Math.round(255*(1+t)),
      B=t<0?255:Math.round(255*(1-t)), G=Math.round(255*(1-Math.abs(t)));
      grc.fillStyle=`rgb(${R},${G},${B})`; } else grc.fillStyle="#ffe066";
    grc.fill(); grc.strokeStyle="#0c0e12"; grc.lineWidth=1.5; grc.stroke();
    if(showIdx){ grc.fillStyle="#0a0a0a"; grc.font="10px monospace";
      grc.textAlign="center"; grc.textBaseline="middle"; grc.fillText(i,P[i][0],P[i][1]); }
  }
  grc.textAlign="left"; grc.textBaseline="alphabetic";
  $("graphhint").textContent =
    `узлов ${last.cnt} · рёбер ${last.edges} · компонент связности ${last.comp} (→ ${last.comp} нулевых мод)`
    + (showIdx?" · номера узлов = строки матрицы":"");
}

//=================== ОТРИСОВКА: собственные значения ===================
const ec=$("eig").getContext("2d"), eW=300,eH=240;
let eigBars=[];
function drawEig(){
  ec.clearRect(0,0,eW,eH); eigBars=[];
  if(!last || last.eigs.length===0){ ec.fillStyle="#9aa0a6"; ec.font="13px monospace";
    ec.fillText("нет мод",10,20); return; }
  const eigs=last.eigs, m=eigs.length, maxv=Math.max(1e-9,...eigs);
  const pad=28, bw=(eW-2*pad)/m, base=eH-26;
  ec.strokeStyle="#2b313c"; ec.beginPath(); ec.moveTo(pad,base); ec.lineTo(eW-6,base); ec.stroke();
  let nzPtr=0;
  for(let i=0;i<m;i++){
    const h=(eigs[i]/maxv)*(base-12);
    const x=pad+i*bw+1, y=base-h, w=Math.max(2,bw-3);
    const isZero = eigs[i]<=1e-6;
    let nzIndex=-1;
    if(!isZero){ nzIndex=nzPtr; nzPtr++; }
    let color;
    if(isZero){ color="#3a4350"; }
    else if(nzIndex < modeMask.length && modeMask[nzIndex]){ color="#ffe066"; }
    else { color="#5a6172"; }
    if(selMode===i){ color="#69db7c"; }
    ec.fillStyle=color;
    ec.fillRect(x,y,w,h);
    eigBars.push({x,y:Math.min(y,base),w,h:Math.max(h,8),idx:i,base,isZero,nzIndex});
  }
  ec.fillStyle="#9aa0a6"; ec.font="11px monospace";
  ec.fillText("λ (по возрастанию) →", pad, eH-8);
  ec.fillText("высота = λ", pad, 14);
}
$("eig").addEventListener("click",e=>{
  const rect=$("eig").getBoundingClientRect();
  const x=(e.clientX-rect.left)*(eW/rect.width), y=(e.clientY-rect.top)*(eH/rect.height);
  for(const b of eigBars){
    if(x>=b.x-1 && x<=b.x+b.w+1 && y>=b.y-6 && y<=b.base){
      if(!b.isZero && b.nzIndex>=0 && b.nzIndex<modeMask.length){
        modeMask[b.nzIndex] = !modeMask[b.nzIndex];
        selMode = b.idx;
      } else {
        selMode = (selMode===b.idx?-1:b.idx);
      }
      recompute(); return;
    }
  }
});

//=================== ОТРИСОВКА: спектр партиалов ===================
const sc=$("spec").getContext("2d"), spW=640,spH=200;
function drawSpec(){
  sc.clearRect(0,0,spW,spH);
  const pad=36, base=spH-28;
  sc.strokeStyle="#2b313c"; sc.beginPath(); sc.moveTo(pad,base); sc.lineTo(spW-6,base); sc.stroke();
  if(!last) return;
  const fs=last.freqs.filter(f=>f>0), as=last.amps.slice(0,fs.length);
  const f0=last.f0;
  const fmax=Math.max(f0*($("n").value)*1.05, ...fs, f0*8)*1.0;
  const X=f=>pad+(f/fmax)*(spW-pad-10);
  // гармонический ряд f0*k (бледные засечки)
  sc.strokeStyle="#33414f"; sc.fillStyle="#5a6172"; sc.font="10px monospace";
  for(let k=1;k*f0<=fmax;k++){ const x=X(k*f0);
    sc.beginPath(); sc.moveTo(x,base); sc.lineTo(x,base+6); sc.stroke();
    if(k<=8||k%2===0) sc.fillText(k+"·f0", x-8, base+18); }
  // наши партиалы
  for(let i=0;i<fs.length;i++){ const x=X(fs[i]); const h=(base-14)*as[i];
    sc.strokeStyle=`rgba(255,224,102,${0.35+0.65*as[i]})`;
    sc.lineWidth=Math.max(1.5,3.2*as[i]);
    sc.beginPath(); sc.moveTo(x,base); sc.lineTo(x,base-h); sc.stroke();
    sc.fillStyle="#ffe066"; sc.beginPath(); sc.arc(x,base-h,2.6,0,7); sc.fill(); }
  // f0 маркер
  const xf=X(f0); sc.strokeStyle="#69db7c"; sc.setLineDash([3,3]);
  sc.beginPath(); sc.moveTo(xf,8); sc.lineTo(xf,base); sc.stroke(); sc.setLineDash([]);
  sc.fillStyle="#69db7c"; sc.font="11px monospace"; sc.fillText("f0",xf-6,8+10);
  sc.fillStyle="#9aa0a6"; sc.fillText("частота, Гц →", pad, spH-8);
}

//=================== КОНВЕЙЕР (текст) ===================
function fmt(x,d=1){ return (Math.round(x*Math.pow(10,d))/Math.pow(10,d)).toString(); }
function drawPipe(){
  if(!last) return;
  const r=last;
  let h="";
  h+=`<b>1.</b> живых клеток: <b>${r.cnt}</b>`;
  if(r.note){ h+=` — <span class="bad">${r.note}</span>`; $("pipe").innerHTML=h; $("table").innerHTML=""; return; }
  h+=`<br><b>2.</b> граф: рёбер <b>${r.edges}</b>, компонент <b>${r.comp}</b>`;
  h+=`<br><b>3.</b> λ = [${r.eigs.map(v=>fmt(v,2)).join(", ")}]`;
  h+=`<br>&nbsp;&nbsp;&nbsp;нулевых (λ≈0): <b>${r.eigs.filter(v=>v<=1e-6).length}</b> = компонент связности ✓`;
  h+=`<br><b>4.</b> √λ ненулевых = [${r.nz.map(v=>fmt(v,3)).join(", ")}]`;
  h+=`<br><b>5.</b> низшая ненулевая → f0: scale = f0/√λ₁ = ${fmt(r.f0)}/${fmt(r.nz[0]??0,3)} = <b>${fmt(r.f0/(r.nz[0]||1),2)}</b>`;
  h+=`<br>&nbsp;&nbsp;&nbsp;все моды-кандидаты (Гц) = [${r.modeFreqs.map(v=>fmt(v,1)).join(", ")}]`;
  h+=`<br><b>6.</b> включённых мод: <b>${r.enabledIdx.length}</b> из ${r.modeFreqs.length}`;
  if(r.enabledIdx.length>0 && r.enabledIdx[0]>0)
    h+=` <span class="muted">(низшая включённая → f0, спектр перемасштабирован)</span>`;
  const fs=r.freqs.filter(f=>f>0);
  h+=`<br>&nbsp;&nbsp;&nbsp;<span style="color:var(--lap)">freqs → [${fs.map(v=>fmt(v,1)).join(", ")}] Гц</span>`;
  h+=`<br>&nbsp;&nbsp;&nbsp;ratios f/f0 = [${fs.map(v=>fmt(v/r.f0,3)).join(", ")}] <span class="muted">(нецелые → инхармонично)</span>`;
  $("pipe").innerHTML=h;
  let t="<table><tr><th>i</th><th>freq, Гц</th><th>f/f0</th><th>amp</th></tr>";
  for(let i=0;i<fs.length;i++){
    t+=`<tr><td>${i+1}</td><td>${fmt(fs[i],1)}</td><td>${fmt(fs[i]/r.f0,3)}</td><td>${fmt(r.amps[i],3)}</td></tr>`;
  }
  t+="</table>";
  $("table").innerHTML=t;
}
