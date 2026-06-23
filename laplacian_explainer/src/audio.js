"use strict";
//=================== WEB AUDIO (аддитивный синтез) ===================
let actx=null, master=null, voices=[], playing=false;
const NVOICE=20;
function initAudio(){
  if(actx) return;
  actx=new (window.AudioContext||window.webkitAudioContext)();
  master=actx.createGain(); master.gain.value=0; master.connect(actx.destination);
  for(let i=0;i<NVOICE;i++){
    const o=actx.createOscillator(); o.type="sine"; o.frequency.value=440;
    const g=actx.createGain(); g.gain.value=0;
    o.connect(g); g.connect(master); o.start();
    voices.push({o,g});
  }
}
function updateAudio(){
  if(!actx||!playing||!last) return;
  const fs=last.freqs, as=last.amps;
  const nf=fs.filter(f=>f>0).length;
  const norm = nf>0? 0.9/ as.slice(0,nf).reduce((a,b)=>a+b,0.0001) : 0; // не клиппить сумму
  const tnow=actx.currentTime;
  for(let i=0;i<NVOICE;i++){ const v=voices[i];
    if(i<nf && fs[i]>0){ v.o.frequency.setTargetAtTime(fs[i],tnow,0.02);
      v.g.gain.setTargetAtTime(as[i]*norm,tnow,0.03); }
    else v.g.gain.setTargetAtTime(0,tnow,0.03);
  }
}
$("play").addEventListener("click",()=>{
  initAudio(); if(actx.state==="suspended") actx.resume();
  playing=!playing;
  const b=$("play");
  if(playing){ b.textContent="■ Стоп"; b.classList.add("on"); b.classList.remove("primary");
    master.gain.setTargetAtTime(0.25,actx.currentTime,0.05); updateAudio(); }
  else { b.textContent="▶ Звук"; b.classList.remove("on"); b.classList.add("primary");
    master.gain.setTargetAtTime(0,actx.currentTime,0.05); }
});
