"use strict";
//=================== ВЫБОР МОД (как _select_modes) ===================
function selectModes(nModes, n, spread){
  if(nModes <= n || spread <= 0) {
    const m = Math.min(n,nModes); return Array.from({length:m},(_,i)=>i);
  }
  const sel = [];
  for(let i=0;i<n;i++){
    const consec = i;
    const spreadIdx = (nModes-1) * (i/((n-1)||1)); // n=1 → 0, как np.linspace(0,nModes-1,1)
    let raw = (1-spread)*consec + spread*spreadIdx;
    sel.push(Math.round(raw));
  }
  sel[0]=0;
  for(let i=1;i<n;i++) if(sel[i] <= sel[i-1]) sel[i]=sel[i-1]+1;
  return sel.filter(x=>x<nModes);
}

//=================== КОМПОНЕНТЫ СВЯЗНОСТИ (BFS) ===================
function components(liveSet, liveList){
  const seen=new Set(); let comp=0;
  const key=(r,c)=>r+","+c;
  for(const [r,c] of liveList){
    if(seen.has(key(r,c))) continue;
    comp++; const st=[[r,c]]; seen.add(key(r,c));
    while(st.length){ const [y,x]=st.pop();
      for(const [dy,dx] of NEIGH){ const ny=y+dy,nx=x+dx;
        if(liveSet.has(key(ny,nx)) && !seen.has(key(ny,nx))){ seen.add(key(ny,nx)); st.push([ny,nx]); } } }
  }
  return comp;
}

//=================== ОСНОВНОЕ ВЫЧИСЛЕНИЕ (map_laplacian + маска мод + shape) ===================
function computeLaplacian(){
  const f0 = +$("f0").value, n = +$("n").value,
        spread = +$("spread").value, alpha = +$("alpha").value,
        shape = +$("shape").value;
  const live=[];
  for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++) if(cells[r][c]) live.push([r,c]);
  const res = {f0,n,spread,alpha,shape, live, cnt:live.length,
    L:null, eigs:[], sqrtAll:[], nz:[], nzOffset:0, modeFreqs:[],
    freqs:new Array(n).fill(0), amps:new Array(n).fill(0),
    vectors:[], comp:0, edges:0, note:"", enabledIdx:[]};

  if(live.length < 2){ res.note="меньше 2 клеток → тишина"; return res; }

  const key=(r,c)=>r+","+c;
  const liveSet=new Set(live.map(([r,c])=>key(r,c)));
  res.comp = components(liveSet, live);
  const pos=new Map(); live.forEach(([r,c],i)=>pos.set(key(r,c),i));
  const cnt=live.length;
  const A=Array.from({length:cnt},()=>Array(cnt).fill(0));
  let edgeCount=0;
  for(let i=0;i<cnt;i++){ const [r,c]=live[i];
    for(const [dr,dc] of NEIGH){ const j=pos.get(key(r+dr,c+dc));
      if(j!==undefined){ A[i][j]=1; } } }
  const L=Array.from({length:cnt},()=>Array(cnt).fill(0));
  for(let i=0;i<cnt;i++){ let deg=0;
    for(let j=0;j<cnt;j++){ if(A[i][j]){ deg++; L[i][j]=-1; } }
    L[i][i]=deg; edgeCount+=deg; }
  res.edges = edgeCount/2;
  res.L=L;
  if(edgeCount===0){ res.note="нет рёбер (изолированные клетки) → тишина"; return res; }

  const {values, vectors} = jacobiEigen(L);
  res.eigs=values; res.vectors=vectors;
  res.sqrtAll = values.map(v=>Math.sqrt(Math.max(v,0)));

  // ненулевые моды и их смещение в отсортированном массиве
  const nzOffset = values.findIndex(v=>v>1e-6);
  if(nzOffset < 0){ res.note="все λ≈0 → тишина"; return res; }
  res.nzOffset = nzOffset;
  const nz = values.slice(nzOffset).map(v=>Math.sqrt(Math.max(v,0)));
  res.nz = nz;

  const baseScale = f0/nz[0];
  const modeFreqs = nz.map(x=>x*baseScale).filter(x=>x<GUARD);
  res.modeFreqs = modeFreqs;
  const mfLen = modeFreqs.length;
  if(mfLen === 0){ res.note="все моды выше guard → тишина"; return res; }

  // перестроить маску если dirty или длина не совпадает (форма сменилась)
  if(maskDirty || modeMask.length !== mfLen){
    const sel0 = selectModes(mfLen, n, spread);
    const selSet = new Set(sel0);
    modeMask = Array.from({length:mfLen},(_,i)=>selSet.has(i));
    maskDirty = false;
  }

  // включённые индексы (в modeFreqs / nz)
  const enabledIdx = modeMask.reduce((acc,v,i)=>{ if(v) acc.push(i); return acc; },[]);
  res.enabledIdx = enabledIdx;
  if(enabledIdx.length === 0) return res;

  // перемасштабирование: низшая включённая мода = f0
  const activeScale = f0 / nz[enabledIdx[0]];
  const num = Math.min(enabledIdx.length, n);
  for(let i=0;i<num;i++) res.freqs[i] = nz[enabledIdx[i]] * activeScale;

  // амплитуды: rolloff 1/(rank+1)^alpha, нормировка max=1
  const rolloff = Array.from({length:num},(_,i)=>1/Math.pow(i+1,alpha));
  const rolloffMx = Math.max(1e-9, rolloff[0]);
  const rolloffN = rolloff.map(v=>v/rolloffMx);

  let amps_raw;
  if(shape <= 0){
    amps_raw = rolloffN;
  } else {
    // краевое возбуждение: степень узла = диагональ L
    const e = live.map((_,j)=>L[j][j]);
    // проекция на собственные векторы включённых мод
    const proj = Array.from({length:num},(_,rank)=>{
      const eigIdx = nzOffset + enabledIdx[rank];
      const phi = vectors[eigIdx]; // собственный вектор (длина cnt)
      let dot=0; for(let j=0;j<cnt;j++) dot += e[j]*phi[j];
      return Math.abs(dot);
    });
    const projMx = Math.max(...proj);
    const projN = projMx <= 1e-9 ? rolloffN.slice() : proj.map(v=>v/projMx);
    amps_raw = Array.from({length:num},(_,i)=>(1-shape)*rolloffN[i]+shape*projN[i]);
  }
  const ampMx = Math.max(1e-9,...amps_raw);
  for(let i=0;i<num;i++) res.amps[i] = amps_raw[i]/ampMx;
  return res;
}
