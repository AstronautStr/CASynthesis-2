// shapes.js — пресеты масок и отрисовка. Глобаль window.SHAPES.
window.SHAPES = (function () {

  const PRESETS = {
    block:  { label: 'Блок 3×3',        rows: ['XXX', 'XXX', 'XXX'] },
    tetro:  { label: 'Т-тетромино',     rows: ['XXX', '.X.'] },
    stick:  { label: 'Палка 1×6',       rows: ['XXXXXX'] },
    ring:   { label: 'Кольцо (1 дыра)', rows: ['XXXX', 'X..X', 'X..X', 'XXXX'] },
    eight:  { label: 'Восьмёрка (2 дыры)', rows: ['XXXXXXX', 'X..X..X', 'X..X..X', 'XXXXXXX'] },
    blob:   { label: 'Клякса',          rows: ['.XX..', 'XXXX.', 'XXXXX', '.XXX.', '..X..'] },
    snake:  { label: 'Змейка',          rows: ['XX...', '.X...', '.XXX.', '...X.', '...XX'] },
  };

  function mask(id) { return LA.maskFromStrings(PRESETS[id].rows); }

  // Отрисовка маски в canvas (вписываем с полями)
  function draw(canvas, m, color = '#7ac6ff') {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const rows = m.length, cols = m[0].length;
    const cell = Math.floor(Math.min((W - 20) / cols, (H - 20) / rows));
    const ox = (W - cell * cols) / 2, oy = (H - cell * rows) / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const x = ox + c * cell, y = oy + r * cell;
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (m[r][c]) {
        ctx.fillStyle = color;
        ctx.fillRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
      }
    }
  }

  // Векторные фигуры для стенда-теста: округлая клякса и колючая звезда
  function drawBlobFigure(canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height, cx = W / 2, cy = H / 2;
    ctx.clearRect(0, 0, W, H);
    const R = Math.min(W, H) * 0.34;
    const pts = [];
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * 2 * Math.PI;
      const r = R * (0.85 + 0.2 * Math.sin(i * 2.7 + 1));
      pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i], n = pts[(i + 1) % pts.length];
      const mx = (p[0] + n[0]) / 2, my = (p[1] + n[1]) / 2;
      if (i === 0) ctx.moveTo(mx, my); else ctx.quadraticCurveTo(p[0], p[1], mx, my);
    }
    const p0 = pts[0], pl = pts[pts.length - 1];
    ctx.quadraticCurveTo(pl[0], pl[1], (pl[0] + p0[0]) / 2, (pl[1] + p0[1]) / 2);
    ctx.closePath();
    ctx.fillStyle = '#9fd8a5'; ctx.fill();
  }

  function drawStarFigure(canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height, cx = W / 2, cy = H / 2;
    ctx.clearRect(0, 0, W, H);
    const R = Math.min(W, H) * 0.42, r = R * 0.36, n = 11;
    ctx.beginPath();
    for (let i = 0; i < n * 2; i++) {
      const a = (i / (n * 2)) * 2 * Math.PI - Math.PI / 2;
      const rr = (i % 2 === 0) ? R : r;
      const x = cx + rr * Math.cos(a), y = cy + rr * Math.sin(a);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = '#e8a4a4'; ctx.fill();
  }

  return { PRESETS, mask, draw, drawBlobFigure, drawStarFigure };
})();
