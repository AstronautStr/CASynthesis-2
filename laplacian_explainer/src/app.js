"use strict";
//=================== ПЕРЕСЧЁТ + ОТРИСОВКА ===================
function recompute(){ last=computeLaplacian();
  if(selMode>=0 && (!last.vectors[selMode])) selMode=-1;
  redrawAll(); updateAudio(); }
function redrawAll(){ drawGrid(); drawGraphView(); drawEig(); drawSpec(); drawPipe(); }

//=================== ВВОД: рисование ===================
let painting=false, paintVal=1;
function cellAt(e){ const rect=$("grid").getBoundingClientRect();
  const x=(e.clientX-rect.left)*(GPX/rect.width), y=(e.clientY-rect.top)*(GPX/rect.height);
  return [Math.floor(y/CELL), Math.floor(x/CELL)]; }
$("grid").addEventListener("mousedown",e=>{ const [r,c]=cellAt(e);
  if(r<0||r>=SIZE||c<0||c>=SIZE) return;
  paintVal = cells[r][c]?0:1; cells[r][c]=paintVal; painting=true; selMode=-1; maskDirty=true; recompute(); });
window.addEventListener("mousemove",e=>{ if(!painting) return; const [r,c]=cellAt(e);
  if(r<0||r>=SIZE||c<0||c>=SIZE) return; if(cells[r][c]!==paintVal){ cells[r][c]=paintVal; maskDirty=true; recompute(); } });
window.addEventListener("mouseup",()=>painting=false);
$("grid").addEventListener("touchstart",e=>{e.preventDefault(); const t=e.touches[0]; const [r,c]=cellAt(t);
  if(r>=0&&r<SIZE&&c>=0&&c<SIZE){ paintVal=cells[r][c]?0:1; cells[r][c]=paintVal; painting=true; selMode=-1; maskDirty=true; recompute(); }},{passive:false});
$("grid").addEventListener("touchmove",e=>{e.preventDefault(); if(!painting)return; const t=e.touches[0]; const [r,c]=cellAt(t);
  if(r>=0&&r<SIZE&&c>=0&&c<SIZE&&cells[r][c]!==paintVal){ cells[r][c]=paintVal; maskDirty=true; recompute(); }},{passive:false});
window.addEventListener("touchend",()=>painting=false);

//=================== КОНТРОЛЫ ===================
for(const id of ["f0","n","spread","alpha","shape"]){
  $(id).addEventListener("input",()=>{
    $("f0v").textContent=$("f0").value; $("nv").textContent=$("n").value;
    $("spreadv").textContent=(+$("spread").value).toFixed(2);
    $("alphav").textContent=(+$("alpha").value).toFixed(2);
    $("shapev").textContent=(+$("shape").value).toFixed(2);
    if(id==="n"||id==="spread") maskDirty=true;
    recompute();
  });
}
$("clear").addEventListener("click",()=>{ cells=Array.from({length:SIZE},()=>Array(SIZE).fill(0)); selMode=-1; maskDirty=true; recompute(); });
$("random").addEventListener("click",()=>{ cells=Array.from({length:SIZE},()=>Array.from({length:SIZE},()=>Math.random()<0.35?1:0)); selMode=-1; maskDirty=true; recompute(); });

//=================== ПРЕСЕТЫ ===================
const PRESETS={
  blinker:[[3,2],[3,3],[3,4]],
  block:[[3,3],[3,4],[4,3],[4,4]],
  beacon:[[2,2],[2,3],[3,2],[4,5],[5,4],[5,5]],
  toad:[[3,3],[3,4],[3,5],[4,2],[4,3],[4,4]],
  lshape:[[2,2],[3,2],[4,2],[4,3],[4,4]],
  ring:[[2,2],[2,3],[2,4],[3,2],[3,4],[4,2],[4,3],[4,4]],
  two:[[1,1],[1,2],[2,1],[5,5],[5,6],[6,5],[6,6]],
  diag:[[1,1],[2,2],[3,3],[4,4],[5,5]],
  plus:[[2,4],[3,4],[4,4],[3,3],[3,5]]
};
$("preset").addEventListener("change",e=>{ const p=PRESETS[e.target.value]; if(!p) return;
  cells=Array.from({length:SIZE},()=>Array(SIZE).fill(0));
  for(const [r,c] of p) if(r<SIZE&&c<SIZE) cells[r][c]=1;
  selMode=-1; maskDirty=true; recompute(); e.target.value=""; });

//=================== СТАРТ ===================
for(const [r,c] of PRESETS.blinker) cells[r][c]=1;  // стартовая форма
recompute();
