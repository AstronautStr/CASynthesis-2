// stations.js — логика всех стендов. Порядок инициализации — внизу файла.
(function () {
  const $ = id => document.getElementById(id);
  const on = (id, ev, fn) => $(id).addEventListener(ev, fn);

  // ==========================================================================
  // Стенд 0 — Тест на себе: тембр ↔ форма (Adeli et al. 2014)
  // ==========================================================================
  function initTest() {
    SHAPES.drawBlobFigure($('t-blob'));
    SHAPES.drawStarFigure($('t-star'));
    on('t-sound1', 'click', () => {
      // резкий: негармонический, яркий, мгновенная атака
      const r = [1, 1.83, 2.77, 4.16, 5.53, 7.11];
      FX.playModal(r.map(x => 330 * x), r.map((_, k) => 1 / Math.pow(k + 1, 0.25)),
        { attack: 0.003, dur: 1.6 });
    });
    on('t-sound2', 'click', () => {
      // мягкий: гармонический, тёмный, медленная атака
      const h = [1, 2, 3, 4, 5, 6];
      FX.playModal(h.map(x => 220 * x), h.map((_, k) => 1 / Math.pow(k + 1, 1.9)),
        { attack: 0.14, dur: 1.9 });
    });
    on('t-reveal', 'click', () => { $('t-answer').style.display = 'block'; });
  }

  // ==========================================================================
  // Стенд 1 — Ритм становится высотой (Штокхаузен)
  // ==========================================================================
  let rpTrain = null;
  function rpRate() {
    // слайдер 0..1 -> 2..400 Гц (логарифмически)
    const x = parseFloat($('rp-rate').value);
    return 2 * Math.pow(200, x);
  }
  function initRhythmPitch() {
    on('rp-toggle', 'click', () => {
      if (rpTrain) { rpTrain.stop(); rpTrain = null; $('rp-toggle').textContent = 'Старт'; return; }
      rpTrain = FX.clickTrain({ rate: rpRate() });
      $('rp-toggle').textContent = 'Стоп';
    });
    on('rp-rate', 'input', () => {
      const r = rpRate();
      $('rp-val').textContent = r.toFixed(1) + ' Гц';
      if (rpTrain) rpTrain.setRate(r);
    });
    $('rp-val').textContent = rpRate().toFixed(1) + ' Гц';
  }

  // ==========================================================================
  // Стенд 2 — Гармония движения (Уитни): отношение темпов двух пульсов
  // ==========================================================================
  let whA = null, whB = null;
  const WH_RATIOS = [
    { label: '3 : 2 (квинта темпов)', v: 3 / 2 },
    { label: '4 : 3', v: 4 / 3 },
    { label: '5 : 4', v: 5 / 4 },
    { label: '7 : 5', v: 7 / 5 },
    { label: '√2 : 1 (иррациональное)', v: Math.SQRT2 },
    { label: 'φ : 1 (золотое, «максимально нерациональное»)', v: (1 + Math.sqrt(5)) / 2 },
  ];
  function whStop() { if (whA) { whA.stop(); whA = null; } if (whB) { whB.stop(); whB = null; } $('wh-toggle').textContent = 'Старт'; }
  function initWhitney() {
    const sel = $('wh-ratio');
    WH_RATIOS.forEach((r, i) => {
      const o = document.createElement('option'); o.value = i; o.textContent = r.label; sel.appendChild(o);
    });
    on('wh-toggle', 'click', () => {
      if (whA) { whStop(); return; }
      const base = parseFloat($('wh-base').value);
      const ratio = WH_RATIOS[parseInt(sel.value)].v;
      whA = FX.clickTrain({ rate: base, burstHz: 1900, pan: -0.6 });
      whB = FX.clickTrain({ rate: base * ratio, burstHz: 900, pan: 0.6 });
      $('wh-toggle').textContent = 'Стоп';
    });
    on('wh-ratio', 'change', () => { if (whA) { whStop(); $('wh-toggle').click(); } });
    on('wh-base', 'input', () => {
      $('wh-base-val').textContent = parseFloat($('wh-base').value).toFixed(1) + ' Гц';
      if (whA) {
        const base = parseFloat($('wh-base').value);
        whA.setRate(base);
        whB.setRate(base * WH_RATIOS[parseInt(sel.value)].v);
      }
    });
  }

  // ==========================================================================
  // Стенд 3 — Синхрония склеивает глаз и ухо (байесовский биндинг)
  // ==========================================================================
  const bind = { running: false, timer: null, nextBeat: 0, flips: [], phase: 0 };
  function bindDraw() {
    const cv = $('bind-canvas'), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const cell = 30, ox = (cv.width - 3 * cell) / 2, oy = (cv.height - 3 * cell) / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
      ctx.strokeRect(ox + c * cell + 0.5, oy + r * cell + 0.5, cell - 1, cell - 1);
    ctx.fillStyle = '#7ac6ff';
    const cells = bind.phase === 0
      ? [[1, 0], [1, 1], [1, 2]]   // горизонтальный блинкер
      : [[0, 1], [1, 1], [2, 1]];  // вертикальный
    cells.forEach(([r, c]) => ctx.fillRect(ox + c * cell + 2, oy + r * cell + 2, cell - 4, cell - 4));
  }
  function bindLoop() {
    if (!bind.running) return;
    const now = FX.now();
    const period = 0.6;
    while (bind.nextBeat < now + 0.15) {
      const lag = parseFloat($('bind-lag').value) / 1000;
      const jitter = $('bind-jitter').checked ? (Math.random() - 0.5) * 0.24 : 0;
      FX.blip(Math.max(0, bind.nextBeat + lag + jitter - now), 780);
      bind.flips.push(bind.nextBeat);
      bind.nextBeat += period;
    }
    while (bind.flips.length && bind.flips[0] <= now) {
      bind.flips.shift(); bind.phase = 1 - bind.phase; bindDraw();
    }
    requestAnimationFrame(bindLoop);
  }
  function initBinding() {
    bindDraw();
    on('bind-toggle', 'click', () => {
      if (bind.running) { bind.running = false; $('bind-toggle').textContent = 'Старт'; return; }
      FX.ensure();
      bind.running = true; bind.nextBeat = FX.now() + 0.2; bind.flips = [];
      $('bind-toggle').textContent = 'Стоп';
      requestAnimationFrame(bindLoop);
    });
    on('bind-lag', 'input', () => { $('bind-lag-val').textContent = $('bind-lag').value + ' мс'; });
  }

  // ==========================================================================
  // Стенд 4 — Материал = профиль затухания (τk = T·(fk/f1)^(−γ))
  // ==========================================================================
  // Кроме профиля τ пресет несёт наклон спектра (tilt: мягкий контакт не
  // возбуждает верхних мод) и шумовой щелчок контакта — без них долгий/короткий
  // звон читается как размер металлического предмета, а не как материал.
  const MAT_PRESETS = {
    metal: { label: 'металл', T: 3.2, g: 0.15, tilt: 0.0,
             noise: { dur: 0.006, cutoff: 3200, type: 'bandpass', q: 1.0, gain: 0.10 } },
    glass: { label: 'стекло', T: 1.1, g: 0.4, tilt: 0.15,
             noise: { dur: 0.007, cutoff: 5200, type: 'bandpass', q: 1.2, gain: 0.22 } },
    stone: { label: 'камень', T: 0.28, g: 0.9, tilt: 0.6,
             noise: { dur: 0.018, cutoff: 2200, type: 'lowpass', q: 0.7, gain: 0.5 } },
    wood:  { label: 'дерево', T: 0.09, g: 1.3, tilt: 1.1,
             noise: { dur: 0.03, cutoff: 750, type: 'lowpass', q: 0.7, gain: 0.6 } },
  };
  let matCur = 'metal';
  function matDrawTaus(taus, freqs) {
    const cv = $('mat-canvas'), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const n = taus.length, bw = cv.width / n;
    const tmax = Math.max(...taus);
    taus.forEach((t, i) => {
      const h = (t / tmax) * (cv.height - 24);
      ctx.fillStyle = '#e0b45c';
      ctx.fillRect(i * bw + 6, cv.height - 14 - h, bw - 12, h);
      ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(Math.round(freqs[i]) + '', i * bw + bw / 2, cv.height - 2);
    });
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'left';
    ctx.fillText('τ мод (высота столбика = время звона), подпись = частота, Гц', 6, 12);
  }
  function matState() {
    const spec = SHAPES.mask($('mat-shape').value);
    const f0 = $('mat-high').checked ? 495 : 165;
    const s = LA.spectrumOf(spec, { f0, nModes: 8 });
    const T = parseFloat($('mat-T').value), g = parseFloat($('mat-g').value);
    const p = MAT_PRESETS[matCur];
    const taus = s.freqs.map(f => T * Math.pow(f / s.freqs[0], -g));
    const amps = s.amps.map((a, i) => a * Math.pow(s.freqs[0] / s.freqs[i], p.tilt));
    return { s, amps, taus, T, g, p };
  }
  function matRefresh() {
    const { s, taus, T, g } = matState();
    $('mat-T-val').textContent = T.toFixed(2) + ' с';
    $('mat-g-val').textContent = g.toFixed(2);
    matDrawTaus(taus, s.freqs);
  }
  function matStrike() {
    const { s, amps, taus, T, p } = matState();
    FX.playModal(s.freqs, amps, { attack: 0.003, taus, dur: Math.min(4, T * 3 + 0.4) });
    FX.noiseBurst(p.noise);
  }
  function initMaterial() {
    const sel = $('mat-shape');
    ['tetro', 'block', 'blob', 'snake'].forEach(id => {
      const o = document.createElement('option'); o.value = id; o.textContent = SHAPES.PRESETS[id].label; sel.appendChild(o);
    });
    Object.entries(MAT_PRESETS).forEach(([id, p]) => {
      $('mat-' + id).addEventListener('click', () => {
        matCur = id;
        Object.keys(MAT_PRESETS).forEach(k => $('mat-' + k).classList.toggle('active', k === id));
        $('mat-T').value = p.T; $('mat-g').value = p.g;
        matRefresh(); matStrike();
      });
    });
    $('mat-metal').classList.add('active');
    on('mat-strike', 'click', matStrike);
    ['mat-T', 'mat-g'].forEach(id => on(id, 'input', matRefresh));
    on('mat-shape', 'change', matRefresh);
    on('mat-high', 'change', matRefresh);
    matRefresh();
  }

  // ==========================================================================
  // Стенд 4½ — Куда ударила жизнь: amp_k = |⟨e, φ_k⟩|, e = точка удара
  // ==========================================================================
  const exc = { mask: null, cells: null, freqs: [], vecs: [], struck: null, mode: null, set: new Set() };
  function excGeom(cv) {
    const rows = exc.mask.length, cols = exc.mask[0].length;
    const cell = Math.floor(Math.min((cv.width - 20) / cols, (cv.height - 20) / rows));
    return { rows, cols, cell, ox: (cv.width - cell * cols) / 2, oy: (cv.height - cell * rows) / 2 };
  }
  function excDraw() {
    const cv = $('exc-canvas'), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const { rows, cols, cell, ox, oy } = excGeom(cv);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const x = ox + c * cell, y = oy + r * cell;
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (exc.mask[r][c]) {
        ctx.fillStyle = exc.mode === 'deg' ? 'rgba(224,180,92,0.55)' : 'rgba(122,198,255,0.55)';
        if (exc.struck && exc.struck[0] === r && exc.struck[1] === c) ctx.fillStyle = '#e0b45c';
        if (exc.set.has(r + ',' + c)) ctx.fillStyle = '#e0b45c';
        ctx.fillRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
      }
    }
  }
  function excBars(amps) {
    const cv = $('exc-bars'), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('амплитуды мод |⟨e, φk⟩|; подпись = частота, Гц (частоты не меняются!)', 6, 12);
    if (!amps) return;
    const n = amps.length, bw = cv.width / n;
    const mx = Math.max(...amps, 1e-9);
    amps.forEach((a, i) => {
      const h = (a / mx) * (cv.height - 34);
      // почти-нулевая мода = удар пришёлся в её узел — красим предупреждающим
      ctx.fillStyle = a / mx < 0.06 ? 'rgba(232,164,164,0.9)' : '#7ac6ff';
      ctx.fillRect(i * bw + 6, cv.height - 14 - h, bw - 12, Math.max(h, 1));
      ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.textAlign = 'center';
      ctx.fillText(Math.round(exc.freqs[i]) + '', i * bw + bw / 2, cv.height - 2);
      ctx.textAlign = 'left';
    });
  }
  function excCompute() {
    exc.mask = SHAPES.mask($('exc-shape').value);
    exc.cells = LA.cellsOf(exc.mask);
    const { values, vectors } = LA.jacobiEig(LA.laplacianOf(exc.mask));
    const modes = [];
    for (let i = 0; i < values.length && modes.length < 8; i++)
      if (values[i] > 1e-9) modes.push(i);
    const l1 = values[modes[0]];
    exc.freqs = modes.map(i => 220 * Math.sqrt(values[i] / l1));
    exc.vecs = modes.map(i => vectors[i]);
    exc.struck = null; exc.mode = null; exc.set.clear();
    excDraw(); excBars(null);
    $('exc-info').textContent = exc.cells.length + ' клеток, ' + exc.freqs.length + ' мод — кликни в клетку';
  }
  function excPlay(e, label) {
    const amps = exc.vecs.map(v => Math.abs(v.reduce((s, x, i) => s + x * e[i], 0)));
    excDraw(); excBars(amps);
    const mx = Math.max(...amps, 1e-9);
    const quiet = amps.filter(a => a / mx < 0.06).length;
    $('exc-info').textContent = label + (quiet ? ` — ${quiet} мод(ы) в узле, молчат` : '');
    const taus = exc.freqs.map(f => 1.3 * Math.pow(f / exc.freqs[0], -0.4));
    FX.playModal(exc.freqs, amps, { attack: 0.004, taus, dur: 2.6 });
  }
  function initExcite() {
    const sel = $('exc-shape');
    ['blob', 'snake', 'ring', 'block', 'tetro'].forEach(id => {
      const o = document.createElement('option'); o.value = id; o.textContent = SHAPES.PRESETS[id].label; sel.appendChild(o);
    });
    on('exc-shape', 'change', excCompute);
    $('exc-canvas').addEventListener('click', ev => {
      const cv = $('exc-canvas'), rect = cv.getBoundingClientRect();
      const x = (ev.clientX - rect.left) * (cv.width / rect.width);
      const y = (ev.clientY - rect.top) * (cv.height / rect.height);
      const { cell, ox, oy } = excGeom(cv);
      const c = Math.floor((x - ox) / cell), r = Math.floor((y - oy) / cell);
      if (!exc.mask[r] || !exc.mask[r][c]) return;
      if ($('exc-multi').checked) {
        // мульти-удар: клики копят набор клеток, звучит по кнопке
        const key = r + ',' + c;
        exc.set.has(key) ? exc.set.delete(key) : exc.set.add(key);
        exc.struck = null; exc.mode = 'multi';
        excDraw();
        $('exc-info').textContent = `набрано клеток: ${exc.set.size} — жми «ударить набор»`;
        return;
      }
      const i0 = exc.cells.findIndex(([rr, cc]) => rr === r && cc === c);
      const e = exc.cells.map((_, i) => (i === i0 ? 1 : 0));
      exc.struck = [r, c]; exc.mode = 'cell';
      excPlay(e, `удар в клетку (${r},${c})`);
    });
    on('exc-hit', 'click', () => {
      if (!exc.set.size) { $('exc-info').textContent = 'набор пуст: включи мульти-удар и кликни клетки'; return; }
      const e = exc.cells.map(([rr, cc]) => (exc.set.has(rr + ',' + cc) ? 1 : 0));
      exc.struck = null; exc.mode = 'multi';
      excPlay(e, `удар в ${exc.set.size} клеток разом (вклады интерферируют)`);
    });
    on('exc-clear', 'click', () => {
      exc.set.clear(); exc.struck = null; excDraw();
      $('exc-info').textContent = 'набор сброшен';
    });
    on('exc-multi', 'change', () => {
      if (!$('exc-multi').checked) { exc.set.clear(); excDraw(); }
    });
    on('exc-deg', 'click', () => {
      const L = LA.laplacianOf(exc.mask);
      const e = L.map((row, i) => row[i]);   // диагональ лапласиана = deg
      exc.struck = null; exc.mode = 'deg';
      excPlay(e, 'e = deg: статичный удар ядра, всегда один и тот же');
    });
    excCompute();
  }

  // ==========================================================================
  // Стенд 5 — Порог слияния: один голос ↔ аккорд
  // ==========================================================================
  function initFusion() {
    const R = [1, 2.13, 3.42, 4.79, 6.28]; // негармонические отношения
    function fusPlay(sync) {
      const h = parseFloat($('fus-h').value);
      const d = parseFloat($('fus-d').value);
      const depth = parseFloat($('fus-depth').value);
      const ratios = R.map(r => (1 - h) * r + h * Math.round(r)); // та же формула, что harm в ядре
      const f0 = 196;
      ratios.forEach((r, k) => {
        FX.playModal([f0 * r], [1], {
          attack: 0.012, dur: 4.5, when: k * d,
          gain: 0.9 / Math.pow(k + 1, 0.5),
          // вибрато вплывает после 1.2 с: склейку слышно как событие (Чоунинг)
          vib: { hz: 5, cents: depth, sync, fadeIn: 1.2 },
        });
      });
    }
    on('fus-h', 'input', () => { $('fus-h-val').textContent = parseFloat($('fus-h').value).toFixed(2); });
    on('fus-d', 'input', () => { $('fus-d-val').textContent = Math.round(parseFloat($('fus-d').value) * 1000) + ' мс'; });
    on('fus-depth', 'input', () => { $('fus-depth-val').textContent = $('fus-depth').value + '¢'; });
    on('fus-play', 'click', () => fusPlay($('fus-sync').checked));
    on('fus-ab', 'click', () => {
      const state = $('fus-ab-state');
      fusPlay(true); state.textContent = 'сейчас: общее…';
      setTimeout(() => { fusPlay(false); state.textContent = 'сейчас: врозь…'; }, 5200);
      setTimeout(() => { state.textContent = ''; }, 10400);
    });
  }

  // ==========================================================================
  // Стенд 6 — Строй существа (кривая диссонанса Сетхареса)
  // ==========================================================================
  const setState = { spec: null, curve: null, mins: [] };
  function setSpectrum() {
    const v = $('set-shape').value;
    if (v === 'harmonic') {
      const f0 = 220, n = 6;
      return {
        freqs: Array.from({ length: n }, (_, k) => f0 * (k + 1)),
        amps: Array.from({ length: n }, (_, k) => 1 / Math.pow(k + 1, 0.7)),
      };
    }
    return LA.spectrumOf(SHAPES.mask(v), { f0: 220, nModes: 6 });
  }
  function setDrawCurve() {
    const cv = $('set-canvas'), ctx = cv.getContext('2d');
    const { curve, mins } = setState;
    ctx.clearRect(0, 0, cv.width, cv.height);
    const padL = 34, padB = 22, W = cv.width - padL - 8, H = cv.height - padB - 10;
    const a0 = curve[0].alpha, a1 = curve[curve.length - 1].alpha;
    const dmax = Math.max(...curve.map(p => p.d)), dmin = Math.min(...curve.map(p => p.d));
    const X = a => padL + (a - a0) / (a1 - a0) * W;
    const Y = d => 10 + (1 - (d - dmin) / (dmax - dmin + 1e-9)) * H;
    // сетка 12-РДО (равномерная темперация) — полутона
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    for (let k = 0; k <= 12; k++) {
      const a = Math.pow(2, k / 12); if (a > a1) break;
      ctx.beginPath(); ctx.moveTo(X(a), 10); ctx.lineTo(X(a), 10 + H); ctx.stroke();
      ctx.fillText(k + '', X(a), cv.height - 8);
    }
    // кривая
    ctx.strokeStyle = '#7ac6ff'; ctx.lineWidth = 2; ctx.beginPath();
    curve.forEach((p, i) => { const x = X(p.alpha), y = Y(p.d); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.lineWidth = 1;
    // минимумы
    mins.forEach((m, i) => {
      ctx.fillStyle = '#e0b45c';
      ctx.beginPath(); ctx.arc(X(m.alpha), Y(m.d), 5, 0, 7); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.fillText((i + 1) + '', X(m.alpha), Y(m.d) - 9);
    });
  }
  function setRefresh() {
    setState.spec = setSpectrum();
    setState.curve = LA.dissonanceCurve(setState.spec.freqs, setState.spec.amps);
    setState.mins = LA.localMinima(setState.curve);
    setDrawCurve();
    const box = $('set-mins'); box.innerHTML = '';
    setState.mins.forEach((m, i) => {
      const b = document.createElement('button');
      const cents = Math.round(1200 * Math.log2(m.alpha));
      b.textContent = `мин.${i + 1}: ×${m.alpha.toFixed(3)} (${cents}¢)`;
      b.addEventListener('click', () => setPlayDyad(m.alpha));
      box.appendChild(b);
    });
  }
  function setPlayDyad(alpha) {
    const { freqs, amps } = setState.spec;
    FX.playModal(freqs, amps, { attack: 0.01, dur: 2.4, pan: -0.35 });
    FX.playModal(freqs.map(f => f * alpha), amps, { attack: 0.01, dur: 2.4, pan: 0.35 });
  }
  function initSethares() {
    const sel = $('set-shape');
    [['harmonic', 'Гармонический спектр (контроль)'], ['tetro', null], ['block', null], ['blob', null], ['stick', null]]
      .forEach(([id, label]) => {
        const o = document.createElement('option');
        o.value = id; o.textContent = label || SHAPES.PRESETS[id].label; sel.appendChild(o);
      });
    on('set-shape', 'change', setRefresh);
    $('set-canvas').addEventListener('click', ev => {
      const cv = $('set-canvas'), rect = cv.getBoundingClientRect();
      const x = (ev.clientX - rect.left) * (cv.width / rect.width);
      const padL = 34, W = cv.width - padL - 8;
      const a0 = setState.curve[0].alpha, a1 = setState.curve[setState.curve.length - 1].alpha;
      const alpha = a0 + Math.min(1, Math.max(0, (x - padL) / W)) * (a1 - a0);
      $('set-click-val').textContent = '×' + alpha.toFixed(3) + ' (' + Math.round(1200 * Math.log2(alpha)) + '¢)';
      setPlayDyad(alpha);
    });
    on('set-tritone', 'click', () => setPlayDyad(Math.pow(2, 6 / 12)));
    on('set-fifth', 'click', () => setPlayDyad(Math.pow(2, 7 / 12)));
    setRefresh();
  }

  // ==========================================================================
  // Стенд 7 — Морфинг спектров: кросс-фейд против оптимального транспорта
  // ==========================================================================
  const ot = { bank: null, A: null, B: null, t: 0, auto: null };
  function otSpectra() {
    ot.A = LA.spectrumOf(SHAPES.mask('block'), { f0: 220, nModes: 5 });
    ot.B = LA.spectrumOf(SHAPES.mask('stick'), { f0: 220, nModes: 5 });
  }
  function otDraw() {
    const cv = $('ot-canvas'), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const fmin = 150, fmax = 2400;
    const X = f => 10 + (Math.log(f / fmin) / Math.log(fmax / fmin)) * (cv.width - 20);
    const line = (f, y0, y1, color, w) => {
      ctx.strokeStyle = color; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(X(f), y0); ctx.lineTo(X(f), y1); ctx.stroke();
    };
    ot.A.freqs.forEach(f => line(f, 8, 28, 'rgba(122,198,255,0.8)', 2));
    ot.B.freqs.forEach(f => line(f, cv.height - 28, cv.height - 8, 'rgba(224,180,92,0.8)', 2));
    const t = ot.t, mode = $('ot-mode').value;
    ctx.fillStyle = '#fff';
    if (mode === 'transport') {
      for (let k = 0; k < 5; k++) {
        const f = Math.pow(ot.A.freqs[k], 1 - t) * Math.pow(ot.B.freqs[k], t);
        line(f, cv.height / 2 - 12, cv.height / 2 + 12, '#9fd8a5', 3);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(X(ot.A.freqs[k]), 28); ctx.lineTo(X(ot.B.freqs[k]), cv.height - 28); ctx.stroke();
      }
    } else {
      ot.A.freqs.forEach(f => line(f, cv.height / 2 - 12, cv.height / 2 + 12, `rgba(122,198,255,${(1 - t).toFixed(2)})`, 3));
      ot.B.freqs.forEach(f => line(f, cv.height / 2 - 12, cv.height / 2 + 12, `rgba(224,180,92,${t.toFixed(2)})`, 3));
    }
    ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('A: блок 3×3', 10, 40);
    ctx.fillText('B: палка 1×6', 10, cv.height - 34);
  }
  function otStart() {
    otStop();
    const a = FX.ensure();
    const mode = $('ot-mode').value;
    const mk = (f, g0) => {
      const osc = a.createOscillator(); osc.type = 'sine'; osc.frequency.value = f;
      const g = a.createGain(); g.gain.value = g0;
      osc.connect(g);
      return { osc, g };
    };
    const out = a.createGain(); out.gain.value = 1;
    const bank = { mode, out, parts: [] };
    const normA = 0.45 / ot.A.amps.reduce((s, x) => s + x, 0);
    const normB = 0.45 / ot.B.amps.reduce((s, x) => s + x, 0);
    if (mode === 'transport') {
      for (let k = 0; k < 5; k++) {
        const p = mk(ot.A.freqs[k], ot.A.amps[k] * normA);
        p.g.connect(out); p.osc.start();
        bank.parts.push(p);
      }
    } else {
      ot.A.freqs.forEach((f, k) => { const p = mk(f, ot.A.amps[k] * normA); p.g.connect(out); p.osc.start(); bank.parts.push(p); });
      ot.B.freqs.forEach((f, k) => { const p = mk(f, 0); p.g.connect(out); p.osc.start(); bank.parts.push(p); });
    }
    out.connect(FX.bus()); // через мастер-громкость и компрессор
    ot.bank = bank;
    otApply();
    $('ot-toggle').textContent = 'Стоп';
  }
  function otApply() {
    if (!ot.bank) { otDraw(); return; }
    const a = FX.ensure(), t = ot.t, C = 0.03;
    const normA = 0.45 / ot.A.amps.reduce((s, x) => s + x, 0);
    const normB = 0.45 / ot.B.amps.reduce((s, x) => s + x, 0);
    if (ot.bank.mode === 'transport') {
      ot.bank.parts.forEach((p, k) => {
        const f = Math.pow(ot.A.freqs[k], 1 - t) * Math.pow(ot.B.freqs[k], t);
        const g = (1 - t) * ot.A.amps[k] * normA + t * ot.B.amps[k] * normB;
        p.osc.frequency.setTargetAtTime(f, a.currentTime, C);
        p.g.gain.setTargetAtTime(g, a.currentTime, C);
      });
    } else {
      ot.bank.parts.forEach((p, i) => {
        const isA = i < 5, k = isA ? i : i - 5;
        const g = isA ? (1 - t) * ot.A.amps[k] * normA : t * ot.B.amps[k] * normB;
        p.g.gain.setTargetAtTime(g, a.currentTime, C);
      });
    }
    otDraw();
  }
  function otStop() {
    if (ot.auto) { cancelAnimationFrame(ot.auto); ot.auto = null; }
    if (ot.bank) {
      const a = FX.ensure();
      ot.bank.parts.forEach(p => { p.g.gain.setTargetAtTime(0, a.currentTime, 0.03); try { p.osc.stop(a.currentTime + 0.2); } catch (e) {} });
      ot.bank = null;
    }
    $('ot-toggle').textContent = 'Старт';
  }
  function initOT() {
    otSpectra(); otDraw();
    on('ot-toggle', 'click', () => (ot.bank ? otStop() : otStart()));
    on('ot-mode', 'change', () => { if (ot.bank) otStart(); else otDraw(); });
    on('ot-t', 'input', () => { ot.t = parseFloat($('ot-t').value); $('ot-t-val').textContent = ot.t.toFixed(2); otApply(); });
    on('ot-auto', 'click', () => {
      if (!ot.bank) otStart();
      const t0 = performance.now(), dur = 5000;
      const step = () => {
        const u = Math.min(1, (performance.now() - t0) / dur);
        ot.t = u; $('ot-t').value = u; $('ot-t-val').textContent = u.toFixed(2);
        otApply();
        if (u < 1) ot.auto = requestAnimationFrame(step); else ot.auto = null;
      };
      ot.auto = requestAnimationFrame(step);
    });
  }

  // ==========================================================================
  // Стенд 8 — Порядок и хаос в статистике спектра (Пуассон против Вигнера)
  // ==========================================================================
  const spc = { rect: [], goe: [] };
  function spcRect() {
    const phi = (1 + Math.sqrt(5)) / 2; // иррациональное отношение сторон²
    const ls = [];
    for (let m = 1; m <= 14; m++) for (let n = 1; n <= 14; n++) ls.push(m * m + phi * n * n);
    ls.sort((a, b) => a - b);
    return ls.slice(2, 50);
  }
  function spcGOE() {
    const n = 44;
    const A = Array.from({ length: n }, () => new Array(n).fill(0));
    const gauss = () => {
      let u = 0, v = 0;
      while (!u) u = Math.random(); while (!v) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    for (let i = 0; i < n; i++) {
      A[i][i] = gauss() * Math.SQRT2;
      for (let j = i + 1; j < n; j++) { A[i][j] = A[j][i] = gauss(); }
    }
    return LA.jacobiEig(A).values.slice(10, 34);
  }
  function spacingsNorm(ls) {
    const s = [];
    for (let i = 0; i + 1 < ls.length; i++) s.push(ls[i + 1] - ls[i]);
    const mean = s.reduce((a, b) => a + b, 0) / s.length;
    return s.map(x => x / mean);
  }
  function spcHist(canvasId, s, curveKind) {
    const cv = $(canvasId), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const bins = 12, smax = 3;
    const h = new Array(bins).fill(0);
    s.forEach(x => { const b = Math.min(bins - 1, Math.floor(x / smax * bins)); h[b]++; });
    const density = h.map(c => c / s.length / (smax / bins));
    const dmax = Math.max(1.05, ...density);
    const W = cv.width - 16, H = cv.height - 26, bw = W / bins;
    density.forEach((d, i) => {
      const bh = d / dmax * H;
      ctx.fillStyle = 'rgba(122,198,255,0.45)';
      ctx.fillRect(8 + i * bw + 1, 8 + H - bh, bw - 2, bh);
    });
    // теоретические кривые: Пуассон e^{−s} и Вигнер (π/2)·s·e^{−πs²/4}
    const draw = (fn, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i <= 120; i++) {
        const x = i / 120 * smax, y = fn(x);
        const px = 8 + x / smax * W, py = 8 + H - Math.min(1, y / dmax) * H;
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke(); ctx.lineWidth = 1;
    };
    draw(x => Math.exp(-x), curveKind === 'poisson' ? '#9fd8a5' : 'rgba(159,216,165,0.35)');
    draw(x => Math.PI / 2 * x * Math.exp(-Math.PI * x * x / 4), curveKind === 'wigner' ? '#e0b45c' : 'rgba(224,180,92,0.35)');
    ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('норм. интервал s между соседними λ (0…3); зелёная = Пуассон, жёлтая = Вигнер', 8, cv.height - 6);
  }
  function spcListen(ls) {
    // окно из 12 соседних λ с минимальным зазором (макс. контраст) → 300..820 Гц
    let best = 0, bestMin = Infinity;
    for (let i = 0; i + 12 <= ls.length; i++) {
      const w = ls.slice(i, i + 12);
      let mn = Infinity;
      for (let k = 0; k + 1 < w.length; k++) mn = Math.min(mn, w[k + 1] - w[k]);
      const span = w[w.length - 1] - w[0];
      const score = mn / span;
      if (score < bestMin) { bestMin = score; best = i; }
    }
    const w = ls.slice(best, best + 12);
    const lo = w[0], hi = w[w.length - 1];
    const freqs = w.map(l => 300 + (l - lo) / (hi - lo) * 520);
    FX.playModal(freqs, freqs.map(() => 1), { attack: 0.05, dur: 4.5 });
  }
  function spcRefresh() {
    spc.rect = spcRect(); spc.goe = spcGOE();
    spcHist('spc-canvas-rect', spacingsNorm(spc.rect), 'poisson');
    spcHist('spc-canvas-goe', spacingsNorm(spc.goe), 'wigner');
  }
  function initSpacings() {
    spcRefresh();
    on('spc-regen', 'click', spcRefresh);
    on('spc-listen-rect', 'click', () => spcListen(spc.rect));
    on('spc-listen-goe', 'click', () => spcListen(spc.goe));
  }

  // ==========================================================================
  // Стенд 9 — Спектр слышит дыры (χ = V − E + F, дыры = C − χ)
  // ==========================================================================
  function holRefresh() {
    const m = SHAPES.mask($('hol-shape').value);
    SHAPES.draw($('hol-canvas'), m, '#9fd8a5');
    const e = LA.eulerOf(m);
    $('hol-report').innerHTML =
      `V=${e.V} клеток, E=${e.E} рёбер, F=${e.F} квадратов &nbsp;→&nbsp; ` +
      `<b>χ = ${e.V}−${e.E}+${e.F} = ${e.chi}</b>; компонент C=${e.C} &nbsp;→&nbsp; ` +
      `<b>дыр β₁ = C − χ = ${e.holes}</b>`;
    return e;
  }
  function initHoles() {
    const sel = $('hol-shape');
    ['blob', 'ring', 'eight', 'block'].forEach(id => {
      const o = document.createElement('option'); o.value = id; o.textContent = SHAPES.PRESETS[id].label; sel.appendChild(o);
    });
    on('hol-shape', 'change', holRefresh);
    on('hol-play', 'click', () => {
      const m = SHAPES.mask($('hol-shape').value);
      const e = LA.eulerOf(m);
      const s = LA.spectrumOf(m, { f0: 330, nModes: 6 });
      FX.playModal(s.freqs, s.amps, { attack: 0.01, dur: 3.0 });
      for (let k = 0; k < e.holes; k++)
        FX.playModal([110 * (1 + 0.007 * k)], [1.6], { attack: 0.25, dur: 3.0, pan: (k ? 0.4 : -0.4) });
    });
    holRefresh();
  }

  // ==========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    const mv = $('master-vol');
    mv.addEventListener('input', () => FX.setMaster(parseFloat(mv.value)));
    document.getElementById('stop-all').addEventListener('click', FX.stopAll);
    initTest(); initRhythmPitch(); initWhitney(); initBinding();
    initMaterial(); initExcite(); initFusion(); initSethares(); initOT();
    initSpacings(); initHoles();
  });
})();
