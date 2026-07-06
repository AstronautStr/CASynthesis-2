// linalg.js — мини-библиотека: метод Якоби для симметричных матриц,
// лапласиан маски клеток, спектр формы, эйлерова характеристика.
// Без зависимостей; глобаль window.LA (конвенция как в laplacian_explainer).
window.LA = (function () {

  // --- Собственные значения/векторы симметричной матрицы (циклический Якоби) ---
  function jacobiEig(Ain) {
    const n = Ain.length;
    const A = Ain.map(row => row.slice());
    const V = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 80; sweep++) {
      let off = 0;
      for (let p = 0; p < n - 1; p++)
        for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-18) break;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = A[p][q];
          if (Math.abs(apq) < 1e-14) continue;
          const theta = (A[q][q] - A[p][p]) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) /
                    (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {           // A <- A·G
            const akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {           // A <- Gᵀ·A
            const apk = A[p][k], aqk = A[q][k];
            A[p][k] = c * apk - s * aqk;
            A[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {           // V <- V·G
            const vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const vals = A.map((r, i) => r[i]);
    const idx = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
    return {
      values: idx.map(i => vals[i]),
      vectors: idx.map(i => V.map(row => row[i])),
    };
  }

  // --- Маска из строк ("X"/"#"/"1" = живая клетка) ---
  function maskFromStrings(rows) {
    return rows.map(r => [...r].map(ch => (ch === 'X' || ch === '#' || ch === '1') ? 1 : 0));
  }

  function cellsOf(mask) {
    const cells = [];
    for (let r = 0; r < mask.length; r++)
      for (let c = 0; c < mask[r].length; c++)
        if (mask[r][c]) cells.push([r, c]);
    return cells;
  }

  // --- Лапласиан графа соседства клеток (8-соседство, как в casynth_core) ---
  function laplacianOf(mask, conn8 = true) {
    const cells = cellsOf(mask);
    const index = new Map(cells.map((rc, i) => [rc[0] + ',' + rc[1], i]));
    const n = cells.length;
    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    const offs = conn8
      ? [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
      : [[-1,0],[1,0],[0,-1],[0,1]];
    cells.forEach(([r, c], i) => {
      offs.forEach(([dr, dc]) => {
        const j = index.get((r + dr) + ',' + (c + dc));
        if (j !== undefined) { L[i][j] = -1; L[i][i] += 1; }
      });
    });
    return L;
  }

  // --- Спектр формы: частоты √(λk/λ1)·f0, амплитуды 1/k^rolloff (как в ядре) ---
  function spectrumOf(mask, { f0 = 220, nModes = 6, rolloff = 0.7, conn8 = true } = {}) {
    const L = laplacianOf(mask, conn8);
    if (L.length === 0) return { freqs: [], amps: [], lambdas: [] };
    const { values } = jacobiEig(L);
    const nz = values.filter(v => v > 1e-9);
    const take = nz.slice(0, nModes);
    if (take.length === 0) return { freqs: [f0], amps: [1], lambdas: [0] };
    const l1 = take[0];
    const freqs = take.map(l => f0 * Math.sqrt(l / l1));
    const amps = take.map((_, k) => 1 / Math.pow(k + 1, rolloff));
    return { freqs, amps, lambdas: take };
  }

  // --- Эйлерова характеристика кубического комплекса (4-соседство!)
  //     V − E + F = χ = C − β1  =>  дыры β1 = C − χ.
  //     Упрощение демо: комплекс из клеток/рёбер/квадратов 2×2 (не 8-связность ядра).
  function eulerOf(mask) {
    const H = mask.length, W = H ? mask[0].length : 0;
    const at = (r, c) => (r >= 0 && r < H && c >= 0 && c < W) ? mask[r][c] : 0;
    let V = 0, E = 0, F = 0;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (!at(r, c)) continue;
      V++;
      if (at(r, c + 1)) E++;
      if (at(r + 1, c)) E++;
      if (at(r, c + 1) && at(r + 1, c) && at(r + 1, c + 1)) F++;
    }
    // компоненты (4-связность), заливкой
    const seen = Array.from({ length: H }, () => new Array(W).fill(false));
    let C = 0;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (!at(r, c) || seen[r][c]) continue;
      C++;
      const stack = [[r, c]]; seen[r][c] = true;
      while (stack.length) {
        const [rr, cc] = stack.pop();
        [[rr-1,cc],[rr+1,cc],[rr,cc-1],[rr,cc+1]].forEach(([nr, nc]) => {
          if (at(nr, nc) && !seen[nr][nc]) { seen[nr][nc] = true; stack.push([nr, nc]); }
        });
      }
    }
    const chi = V - E + F;
    return { V, E, F, C, chi, holes: C - chi };
  }

  // --- Кривая диссонанса Сетхареса (модель Пломпа–Левельта) ---
  // d пары = min(a1,a2)·(e^(−3.5·s·Δf) − e^(−5.75·s·Δf)),  s = 0.24/(0.021·fmin + 19)
  function pairDissonance(f1, a1, f2, a2) {
    const fmin = Math.min(f1, f2), df = Math.abs(f2 - f1);
    const s = 0.24 / (0.021 * fmin + 19);
    return Math.min(a1, a2) * (Math.exp(-3.5 * s * df) - Math.exp(-5.75 * s * df));
  }

  // суммарный диссонанс объединённого спектра {S, α·S}
  function dissonanceCurve(freqs, amps, alphaMin = 1.0, alphaMax = 2.1, steps = 440) {
    const curve = [];
    for (let i = 0; i <= steps; i++) {
      const alpha = alphaMin + (alphaMax - alphaMin) * (i / steps);
      const F = freqs.concat(freqs.map(f => f * alpha));
      const A = amps.concat(amps);
      let d = 0;
      for (let p = 0; p < F.length - 1; p++)
        for (let q = p + 1; q < F.length; q++)
          d += pairDissonance(F[p], A[p], F[q], A[q]);
      curve.push({ alpha, d });
    }
    return curve;
  }

  function localMinima(curve, skipBelow = 1.02) {
    const mins = [];
    for (let i = 1; i < curve.length - 1; i++) {
      const { alpha, d } = curve[i];
      if (alpha < skipBelow) continue;
      if (d <= curve[i - 1].d && d <= curve[i + 1].d &&
          (curve[i - 1].d - d > 1e-6 || curve[i + 1].d - d > 1e-6))
        mins.push({ alpha, d, i });
    }
    mins.sort((a, b) => a.d - b.d);
    return mins.slice(0, 6).sort((a, b) => a.alpha - b.alpha);
  }

  return { jacobiEig, maskFromStrings, cellsOf, laplacianOf, spectrumOf,
           eulerOf, pairDissonance, dissonanceCurve, localMinima };
})();
