// 滑块展示目标末帧钳位和软终止，小屏默认收起补充说明。
// 初始读数和图形写在 HTML 中，离线与静态海报都能读到完整示例。
// 控件只更新所在图的数据，键盘操作沿用浏览器原生行为。

// Pages 只托管 docs。在线阅读时到 GitHub 浏览源码和 Markdown，本地阅读沿用相对路径。
if (location.hostname === "jasong98.github.io") {
  const source = "https://github.com/JasonG98/nanomimic/blob/main/docs/wiki/html/";
  for (const link of document.querySelectorAll('a[href^="../"]')) {
    const href = link.getAttribute("href");
    const url = new URL(href, source);
    if (href.startsWith("../../../") || url.pathname.endsWith(".md")) {
      if (url.pathname.endsWith("/")) url.pathname = url.pathname.replace("/blob/", "/tree/");
      link.href = url.href;
    }
  }
}

function enableFrameWindow(panel) {
  const slider = panel.querySelector("input");
  const current = panel.querySelector("#frame-value");
  const marker = panel.querySelector("#frame-marker");
  const slots = panel.querySelectorAll(".frame-list li");
  const end = Number(slider.max);
  function draw() {
    const start = Number(slider.value);
    current.value = String(start);
    marker.style.setProperty("--position", (100 * start / end) + "%");
    slots.forEach((slot, index) => {
      const frame = start + 5 * index;
      slot.textContent = String(Math.min(frame, end));
      slot.classList.toggle("held", frame > end);
    });
    slider.setAttribute("aria-valuetext", "当前第 " + start + " 帧");
  }
  slider.addEventListener("input", draw);
  draw();
  slider.parentElement.hidden = false;
}

function enableRiskCurve(panel) {
  const slider = panel.querySelector("input");
  const error = panel.querySelector("#risk-value");
  const risk = panel.querySelector("#risk-prob");
  const survival = panel.querySelector("#risk-survive");
  const point = panel.querySelector("#risk-point");
  const minimum = Number(slider.min);
  const maximum = Number(slider.max);
  function draw() {
    const value = Number(slider.value);
    const probability = 1 / (1 + Math.exp(-(value - 1) / 0.05));
    error.value = value.toFixed(2);
    risk.value = (100 * probability).toFixed(1) + "%";
    survival.value = (100 * (1 - probability)).toFixed(1) + "%";
    point.setAttribute("cx", String(57 + 411 * (value - minimum) / (maximum - minimum)));
    point.setAttribute("cy", String(151 - 126 * probability));
    slider.setAttribute("aria-valuetext", "归一化误差 " + error.value + "，终止概率 " + risk.value);
  }
  slider.addEventListener("input", draw);
  draw();
  slider.parentElement.hidden = false;
}

const frameDemo = document.getElementById("frame-demo");
if (frameDemo) enableFrameWindow(frameDemo);
const riskDemo = document.getElementById("risk-demo");
if (riskDemo) enableRiskCurve(riskDemo);

const notes = [...document.querySelectorAll(".extra-note")];
const narrow = window.matchMedia("(max-width: 700px)");
function arrangeNotes() {
  for (const note of notes) note.open = !narrow.matches;
}
arrangeNotes();
narrow.addEventListener("change", arrangeNotes);

// 打印时展开全部说明，关闭打印预览后恢复读者的展开选择。
let openBeforePrint = [];
window.addEventListener("beforeprint", () => {
  openBeforePrint = notes.map(note => note.open);
  for (const note of notes) note.open = true;
});
window.addEventListener("afterprint", () => {
  notes.forEach((note, index) => { note.open = openBeforePrint[index]; });
});
