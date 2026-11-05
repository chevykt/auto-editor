// Installed in every page the recorder opens. The real mouse pointer is never
// captured, so this draws one, and blurs whatever the plan says is private.
// The runner fills __TUTORIAL_BLUR__ before installing it.
(() => {
  if (window.top !== window || window.__cursor) return;

  const blur = __TUTORIAL_BLUR__;
  const key = "__tutorial_cursor";
  const saved = JSON.parse(sessionStorage.getItem(key) || "null");

  const cursor = {
    // a new page starts the pointer where the last one left it
    x: saved ? saved.x : innerWidth / 2,
    y: saved ? saved.y : innerHeight / 2,
    el: null,

    place() {
      if (this.el) this.el.style.transform = `translate(${this.x}px, ${this.y}px)`;
      sessionStorage.setItem(key, JSON.stringify({ x: this.x, y: this.y }));
    },

    moveTo(x, y, ms) {
      const fromX = this.x, fromY = this.y, began = performance.now();
      return new Promise((done) => {
        const frame = (now) => {
          const t = Math.min(1, (now - began) / ms);
          const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
          this.x = fromX + (x - fromX) * ease;
          this.y = fromY + (y - fromY) * ease;
          this.place();
          if (t < 1) requestAnimationFrame(frame); else done();
        };
        requestAnimationFrame(frame);
      });
    },

    click() {
      const ring = document.createElement("div");
      ring.style.cssText =
        "position:fixed;z-index:2147483646;pointer-events:none;border-radius:50%;" +
        "border:3px solid rgba(91,76,224,.9);width:12px;height:12px;" +
        `left:${this.x - 6}px;top:${this.y - 6}px;transition:all .4s ease-out`;
      document.documentElement.appendChild(ring);
      requestAnimationFrame(() => {
        ring.style.transform = "scale(4)";
        ring.style.opacity = "0";
      });
      setTimeout(() => ring.remove(), 450);
    },
  };
  window.__cursor = cursor;

  const install = () => {
    if (blur.length) {
      const style = document.createElement("style");
      style.textContent = `${blur.join(",")}{filter:blur(8px) !important}`;
      document.documentElement.appendChild(style);
    }
    const el = document.createElement("div");
    el.style.cssText =
      "position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;" +
      "width:24px;height:24px;margin:-2px 0 0 -3px";
    el.innerHTML =
      '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M3 2l15 11-6.5 1 ' +
      '3.5 7-3 1.5-3.5-7L3 20z" fill="#111" stroke="#fff" stroke-width="1.5"/></svg>';
    document.documentElement.appendChild(el);
    cursor.el = el;
    cursor.place();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
