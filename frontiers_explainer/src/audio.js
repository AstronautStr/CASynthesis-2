// audio.js — общий WebAudio-слой. Глобаль window.FX.
// Контекст создаётся лениво (по первому клику — политика автоплея браузеров).
window.FX = (function () {
  let ctx = null, master = null;
  let masterVol = 0.25;
  const liveHandles = new Set();

  function ensure() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      master = ctx.createGain();
      master.gain.value = masterVol;
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -14; comp.ratio.value = 8;
      comp.attack.value = 0.003; comp.release.value = 0.2;
      master.connect(comp); comp.connect(ctx.destination);
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }

  function setMaster(v) { masterVol = v; if (master) master.gain.value = v; }

  function stopAll() { [...liveHandles].forEach(h => { try { h.stop(); } catch (e) {} }); }

  // --- Аддитивный «модальный» голос ---------------------------------------
  // playModal(freqs, amps, opts):
  //   attack (с), dur (с), taus[] — пер-модовые постоянные экспон. затухания
  //   (null = сустейн до dur), when (с от сейчас), pan [-1..1],
  //   vib {hz, cents, sync} — вибрато, общее или независимое по модам.
  function playModal(freqs, amps, opts = {}) {
    const a = ensure();
    const { attack = 0.008, dur = 2.0, taus = null, when = 0, pan = 0, vib = null, gain = 1 } = opts;
    const t0 = a.currentTime + 0.03 + when;
    const norm = gain * 0.5 / Math.max(1e-6, amps.reduce((s, x) => s + x, 0));
    const out = a.createGain(); out.gain.value = 1;
    let panNode = null;
    if (a.createStereoPanner) { panNode = a.createStereoPanner(); panNode.pan.value = pan; }
    (panNode ? out.connect(panNode) : null);
    (panNode ? panNode : out).connect(master);

    const nodes = [];
    let sharedLfo = null, sharedDepthPerOsc = [];
    if (vib && vib.sync) {
      sharedLfo = a.createOscillator();
      sharedLfo.frequency.value = vib.hz;
      sharedLfo.start(t0); sharedLfo.stop(t0 + dur + 0.5);
      nodes.push(sharedLfo);
    }

    freqs.forEach((f, k) => {
      if (f <= 0 || f > 12000) return;
      const osc = a.createOscillator(); osc.type = 'sine'; osc.frequency.value = f;
      const g = a.createGain(); g.gain.value = 0;
      osc.connect(g); g.connect(out);
      const peak = amps[k] * norm;
      g.gain.setValueAtTime(0, t0);
      g.gain.linearRampToValueAtTime(peak, t0 + attack);
      if (taus && taus[k]) {
        g.gain.setTargetAtTime(0, t0 + attack, taus[k]);
        // мягкий финальный спад, чтобы stop() не щёлкал на недозвеневшем хвосте
        g.gain.setTargetAtTime(0, t0 + dur, 0.03);
      } else {
        g.gain.setValueAtTime(peak, t0 + Math.max(attack, dur - 0.09));
        g.gain.linearRampToValueAtTime(0, t0 + dur);
      }
      if (vib) {
        const depth = a.createGain();
        const dv = f * (Math.pow(2, vib.cents / 1200) - 1);
        if (vib.fadeIn) {
          // трюк Чоунинга: вибрато вплывает после fadeIn секунд — склейка
          // партиалов слышна как событие, а не как стационарное качество
          depth.gain.setValueAtTime(0, t0);
          depth.gain.setValueAtTime(0, t0 + vib.fadeIn);
          depth.gain.linearRampToValueAtTime(dv, t0 + vib.fadeIn + 0.8);
        } else {
          depth.gain.value = dv;
        }
        let lfo = sharedLfo;
        if (!vib.sync) {
          lfo = a.createOscillator();
          lfo.frequency.value = vib.hz * (0.65 + 0.7 * Math.random());
          // случайная фаза на полный круг: старт со сдвигом до целого периода
          lfo.start(t0 + Math.random() / vib.hz); lfo.stop(t0 + dur + 0.5);
          nodes.push(lfo);
        }
        lfo.connect(depth); depth.connect(osc.frequency);
      }
      osc.start(t0); osc.stop(t0 + dur + 0.3);
      nodes.push(osc);
    });

    const handle = {
      stop() {
        const t = a.currentTime;
        out.gain.setTargetAtTime(0, t, 0.03);
        nodes.forEach(n => { try { n.stop(t + 0.25); } catch (e) {} });
        liveHandles.delete(handle);
      }
    };
    liveHandles.add(handle);
    setTimeout(() => liveHandles.delete(handle), (when + dur + 0.6) * 1000);
    return handle;
  }

  // --- Пульс-трейн: антиалиасная пила через полосовой «резонатор» -----------
  // Скачок пилы раз в период = щелчок; фильтр придаёт ему звон burstHz.
  // Осцилляторы WebAudio band-limited на любой частоте, поэтому глиссандо
  // «ритм → высота» чистое (ресемплируемый буфер на больших rate алиасил в шум).
  function clickTrain({ burstHz = 1900, rate = 4, pan = 0, gain = 1 } = {}) {
    const a = ensure();
    const osc = a.createOscillator(); osc.type = 'sawtooth'; osc.frequency.value = rate;
    const bp = a.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.value = burstHz; bp.Q.value = 3;
    const g = a.createGain();
    // энергия трейна растёт ~√rate — компенсируем, чтобы громкость не плыла
    const lvl = r => gain * Math.min(1.3, 2.0 / Math.sqrt(Math.max(1, r)));
    g.gain.value = lvl(rate);
    osc.connect(bp); bp.connect(g);
    let panNode = null;
    if (a.createStereoPanner) { panNode = a.createStereoPanner(); panNode.pan.value = pan; g.connect(panNode); panNode.connect(master); }
    else g.connect(master);
    // скачок пилы — через полпериода после старта; сдвигаем старт так, чтобы
    // щелчки трейнов, созданных рядом по времени, легли на общую сетку 0.25 с
    // (полиритмам стенда Уитни нужны совпадения ударов)
    let ts = Math.ceil((a.currentTime + 0.08) / 0.25) * 0.25 - 0.5 / rate;
    while (ts < a.currentTime + 0.02) ts += 1 / rate;
    osc.start(ts);
    const handle = {
      setRate(r) {
        osc.frequency.setTargetAtTime(r, a.currentTime, 0.02);
        g.gain.setTargetAtTime(lvl(r), a.currentTime, 0.05);
      },
      stop() { try { g.gain.setTargetAtTime(0, a.currentTime, 0.02); osc.stop(a.currentTime + 0.15); } catch (e) {} liveHandles.delete(handle); }
    };
    liveHandles.add(handle);
    return handle;
  }

  // --- Шумовой транзиент контакта (удар молоточка/пальца) -------------------
  // Короткий фильтрованный шум; t0 совпадает с playModal — «удар» синхронен модам.
  function noiseBurst({ when = 0, dur = 0.025, cutoff = 1500, type = 'lowpass', q = 0.7, gain = 0.4, pan = 0 } = {}) {
    const a = ensure();
    const t0 = a.currentTime + 0.03 + when;
    const n = Math.max(64, Math.floor(a.sampleRate * dur * 4));
    const buf = a.createBuffer(1, n, a.sampleRate);
    const d = buf.getChannelData(0);
    const tau = a.sampleRate * dur / 3;
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / tau);
    const src = a.createBufferSource(); src.buffer = buf;
    const f = a.createBiquadFilter(); f.type = type; f.frequency.value = cutoff; f.Q.value = q;
    const g = a.createGain(); g.gain.value = gain;
    src.connect(f); f.connect(g);
    if (a.createStereoPanner) { const p = a.createStereoPanner(); p.pan.value = pan; g.connect(p); p.connect(master); }
    else g.connect(master);
    src.start(t0);
  }

  // --- Короткий перкуссивный блип (для стенда синхронии) --------------------
  function blip(when = 0, f = 880) {
    const a = ensure();
    const t0 = a.currentTime + when;
    const osc = a.createOscillator(); osc.frequency.value = f;
    const g = a.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(0.6, t0 + 0.003);
    g.gain.setTargetAtTime(0, t0 + 0.003, 0.03);
    osc.connect(g); g.connect(master);
    osc.start(t0); osc.stop(t0 + 0.25);
  }

  function now() { return ensure().currentTime; }

  function bus() { ensure(); return master; }

  return { ensure, setMaster, stopAll, playModal, clickTrain, noiseBurst, blip, now, bus };
})();
