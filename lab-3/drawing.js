// No network assets. The Python widget synchronizes only drawing state.
export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "trigger-editor";
    root.innerHTML = `
      <div class="tools">
        <label>Draw on digit <select aria-label="Drawing digit"></select></label>
        <label>Alpha <input aria-label="Trigger alpha" type="range" min="0" max="1" step="0.05"><output></output></label>
      </div>
      <div class="tools">
        <label>Brush <select aria-label="Brush size"><option value="1">1 pixel</option><option value="2">2 pixels</option><option value="3">3 pixels</option></select></label>
        <button type="button" aria-pressed="false">Eraser</button>
        <button type="button">Undo</button>
        <button type="button">Clear</button>
        <label><input aria-label="Show pixel grid" type="checkbox" checked> Grid</label>
      </div>
      <canvas width="336" height="336" role="img" aria-label="Draw your trigger on the digit"></canvas>
      <p class="status" aria-live="polite"></p>
      <p class="status">Alpha applies to new strokes. Lift the pen to update code and predictions. Coordinates start at 0.</p>`;
    el.appendChild(root);
    const digitSelect = root.querySelector('[aria-label="Drawing digit"]');
    const alpha = root.querySelector('[aria-label="Trigger alpha"]');
    const alphaOutput = root.querySelector("output");
    const brush = root.querySelector('[aria-label="Brush size"]');
    const grid = root.querySelector('[aria-label="Show pixel grid"]');
    const [eraser, undo, clear] = root.querySelectorAll("button");
    const canvas = root.querySelector("canvas");
    const status = root.querySelector(".status");
    const ctx = canvas.getContext("2d");
    const clone = value => value.map(row => [...row]);
    let mask = clone(model.get("mask"));
    let opacity = model.get("opacity");
    let digit = model.get("digit");
    let drawing = false, erase = false, last = null, publishing = false;
    let strokeMask = null, strokeAlpha = opacity;
    const history = [];
    for (let i = 0; i < model.get("images").length; i++) digitSelect.add(new Option(String(i), String(i)));
    digitSelect.value = String(digit);
    alpha.value = String(opacity);

    function render() {
      const images = model.get("images");
      const image = images[digit];
      const scale = canvas.width / 28;
      for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) {
        const a = mask[r][c];
        const gray = Math.round(255 * (1 - ((1 - a) * image[r][c] + a)));
        ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
        ctx.fillRect(c * scale, r * scale, scale, scale);
      }
      if (grid.checked) {
        ctx.strokeStyle = "rgba(100,116,139,.18)";
        ctx.lineWidth = .5;
        ctx.beginPath();
        for (let i = 0; i <= 28; i++) {
          ctx.moveTo(i * scale, 0); ctx.lineTo(i * scale, canvas.height);
          ctx.moveTo(0, i * scale); ctx.lineTo(canvas.width, i * scale);
        }
        ctx.stroke();
      }
      alphaOutput.value = opacity.toFixed(2);
      const count = mask.flat().filter(v => v > 0).length;
      status.textContent = `${count} drawn pixels · brush alpha ${opacity.toFixed(2)} · ${erase ? "erasing" : "drawing"}`;
      undo.disabled = history.length === 0;
    }
    function remember() {
      history.push(clone(mask));
      if (history.length > 30) history.shift();
    }
    function publish() {
      publishing = true;
      model.set("mask", clone(mask));
      model.set("opacity", opacity);
      model.set("digit", digit);
      model.save_changes();
      publishing = false;
    }
    function point(event) {
      const rect = canvas.getBoundingClientRect();
      return [Math.floor((event.clientY - rect.top) * 28 / rect.height),
              Math.floor((event.clientX - rect.left) * 28 / rect.width)];
    }
    function stamp(r, c) {
      const size = Number(brush.value);
      const start = -Math.floor((size - 1) / 2);
      for (let dr = start; dr < start + size; dr++) for (let dc = start; dc < start + size; dc++) {
        const row = r + dr, col = c + dc;
        if (row >= 0 && row < 28 && col >= 0 && col < 28) {
          // Blend once per stroke, regardless of pointer-event frequency.
          mask[row][col] = erase ? 0 : 1 - (1 - strokeMask[row][col]) * (1 - strokeAlpha);
        }
      }
    }
    function paint(event) {
      const [r, c] = point(event);
      const [r0, c0] = last || [r, c];
      const steps = Math.max(Math.abs(r - r0), Math.abs(c - c0), 1);
      for (let i = 0; i <= steps; i++) {
        stamp(Math.round(r0 + (r - r0) * i / steps), Math.round(c0 + (c - c0) * i / steps));
      }
      last = [r, c];
      render();
      status.textContent += ` · row ${Math.max(0, Math.min(27, r))}, column ${Math.max(0, Math.min(27, c))}`;
    }
    canvas.onpointerdown = event => {
      if (event.button !== 0) return;
      event.preventDefault();
      canvas.setPointerCapture(event.pointerId);
      remember(); strokeMask = clone(mask); strokeAlpha = opacity;
      drawing = true; last = null; paint(event);
    };
    canvas.onpointermove = event => {
      if (drawing) paint(event);
      else {
        const [r, c] = point(event);
        render(); status.textContent += ` · row ${r}, column ${c}`;
      }
    };
    const finish = () => { if (drawing) { drawing = false; last = null; strokeMask = null; publish(); } };
    canvas.onpointerup = finish;
    canvas.onpointercancel = finish;
    canvas.onlostpointercapture = finish;
    alpha.oninput = () => {
      opacity = Number(alpha.value); render(); publish();
    };
    alpha.onchange = publish;
    digitSelect.onchange = () => { digit = Number(digitSelect.value); render(); publish(); };
    grid.onchange = render;
    eraser.onclick = () => { erase = !erase; eraser.setAttribute("aria-pressed", String(erase)); render(); };
    undo.onclick = () => { if (history.length) { mask = history.pop(); render(); publish(); } };
    clear.onclick = () => { remember(); mask = Array.from({ length: 28 }, () => Array(28).fill(0)); render(); publish(); };
    const sync = () => {
      if (publishing) return;
      if (!drawing) mask = clone(model.get("mask"));
      opacity = model.get("opacity"); digit = model.get("digit");
      alpha.value = String(opacity); digitSelect.value = String(digit); render();
    };
    model.on("change:mask", sync);
    model.on("change:opacity", sync);
    model.on("change:digit", sync);
    render();
    return () => {
      model.off("change:mask", sync); model.off("change:opacity", sync); model.off("change:digit", sync);
      root.remove();
    };
  },
};
