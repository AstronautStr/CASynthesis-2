"use strict";
//=================== ЛИНЕЙНАЯ АЛГЕБРА: симметричный Jacobi ===================
// Возвращает {values:[], vectors:[]} отсортированные по возрастанию λ.
// vectors[k] — собственный вектор для values[k] (длина n).
function jacobiEigen(Ain){
  const n = Ain.length;
  if(n===0) return {values:[],vectors:[]};
  if(n===1) return {values:[Ain[0][0]], vectors:[[1]]};
  const a = Ain.map(r=>r.slice());
  const v = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
  for(let sweep=0; sweep<100; sweep++){
    let off=0;
    for(let p=0;p<n;p++) for(let q=p+1;q<n;q++) off += a[p][q]*a[p][q];
    if(off < 1e-22) break;
    for(let p=0;p<n;p++){
      for(let q=p+1;q<n;q++){
        const apq=a[p][q];
        if(Math.abs(apq) < 1e-18) continue;
        const denom = a[q][q]-a[p][p];
        let t;
        if(Math.abs(denom) < 1e-30){ t = 1; }
        else { const theta = denom/(2*apq);
               t = (theta>=0?1:-1)/(Math.abs(theta)+Math.sqrt(theta*theta+1)); }
        const c = 1/Math.sqrt(t*t+1), s = t*c;
        for(let k=0;k<n;k++){ const akp=a[k][p], akq=a[k][q];
          a[k][p]=c*akp - s*akq; a[k][q]=s*akp + c*akq; }
        for(let k=0;k<n;k++){ const apk=a[p][k], aqk=a[q][k];
          a[p][k]=c*apk - s*aqk; a[q][k]=s*apk + c*aqk; }
        for(let k=0;k<n;k++){ const vkp=v[k][p], vkq=v[k][q];
          v[k][p]=c*vkp - s*vkq; v[k][q]=s*vkp + c*vkq; }
      }
    }
  }
  const pairs=[];
  for(let i=0;i<n;i++) pairs.push({val:a[i][i], vec:v.map(r=>r[i])});
  pairs.sort((x,y)=>x.val-y.val);
  return {values:pairs.map(p=>p.val), vectors:pairs.map(p=>p.vec)};
}
