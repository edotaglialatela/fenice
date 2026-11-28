// Fenice documentation: search, the page list on small screens, copy buttons on code blocks,
// and the highlight of the current section in "On this page". Pages work without it.
(function () {
  "use strict";
  const root = document.body.dataset.root || "";
  const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const typing = () => {
    const el = document.activeElement;
    return el && (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || el.isContentEditable);
  };

  // ---- Search: results drop down under the box; arrows move, Enter opens, Escape closes, "/" focuses.
  const box = document.getElementById("fv-search"), panel = document.getElementById("fv-results");
  if (box && panel) {
    let hits = [], active = -1;
    const show = on => { panel.hidden = !on; box.setAttribute("aria-expanded", String(on)); };
    const render = () => {
      panel.innerHTML = hits.length ? hits.map((e, i) =>
        `<a href="${root}${e.url}" role="option" id="fv-hit-${i}"${i === active ? ' aria-selected="true"' : ""}>` +
        `<strong>${esc(e.heading || e.title)}</strong>` + (e.heading ? `<small>${esc(e.title)}</small>` : "") +
        (e.text ? `<span>${esc(e.text.slice(0, 140))}</span>` : "") + "</a>").join("")
        : `<p class="none">No page mentions “${esc(box.value.trim())}”.</p>`;
      if (active >= 0) {
        box.setAttribute("aria-activedescendant", `fv-hit-${active}`);
        document.getElementById(`fv-hit-${active}`).scrollIntoView({ block: "nearest" });
      } else {
        box.removeAttribute("aria-activedescendant");
      }
    };
    box.addEventListener("input", () => {
      const q = box.value.trim().toLowerCase();
      if (!q) { show(false); return; }
      const terms = q.split(/\s+/);
      const all = window.FV_SEARCH || [];
      const found = all.filter(e => terms.every(t => (e.title + " " + e.heading + " " + e.text).toLowerCase().includes(t)));
      // Page titles holding every term first, then section headings, then API members, then the rest.
      const rank = e => {
        if (!terms.every(t => (e.heading || e.title).toLowerCase().includes(t))) return 3;
        if (!e.heading) return 0;
        return e.url.startsWith("api/") ? 2 : 1;
      };
      hits = found.map((e, i) => [rank(e), i, e]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(x => x[2]).slice(0, 12);
      active = hits.length ? 0 : -1;
      render();
      show(true);
    });
    box.addEventListener("keydown", ev => {
      if ((ev.key === "ArrowDown" || ev.key === "ArrowUp") && hits.length && !panel.hidden) {
        ev.preventDefault();
        active = (active + (ev.key === "ArrowDown" ? 1 : hits.length - 1)) % hits.length;
        render();
      } else if (ev.key === "Enter" && hits.length && !panel.hidden) {
        ev.preventDefault();
        location.href = root + hits[Math.max(active, 0)].url;
      } else if (ev.key === "Escape") {
        box.value = "";
        show(false);
      }
    });
    box.addEventListener("focus", () => { if (box.value.trim()) show(true); });
    document.addEventListener("click", ev => { if (!ev.target.closest(".search")) show(false); });
    document.addEventListener("keydown", ev => {
      if (ev.key === "/" && !typing() && !ev.ctrlKey && !ev.metaKey && !ev.altKey) { ev.preventDefault(); box.focus(); }
    });
  }

  // ---- Page list: a drawer on small screens.
  const menu = document.querySelector(".menu-btn"), side = document.getElementById("sidebar");
  if (menu && side) {
    const isOpen = () => document.body.classList.contains("nav-open");
    const set = on => { document.body.classList.toggle("nav-open", on); menu.setAttribute("aria-expanded", String(on)); };
    menu.addEventListener("click", () => set(!isOpen()));
    document.addEventListener("keydown", ev => { if (ev.key === "Escape" && isOpen()) { set(false); menu.focus(); } });
    document.addEventListener("click", ev => { if (isOpen() && !ev.target.closest("#sidebar, .menu-btn")) set(false); });
    const current = side.querySelector('[aria-current="page"]');
    if (current) side.scrollTop = current.offsetTop - side.clientHeight / 3;   // the list scrolls, not the page
  }

  // ---- Copy buttons on code blocks (not inside the interactive explorers).
  const ICON = '<svg class="i-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/>' +
    '<path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg><svg class="i-check" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  const timers = new WeakMap();
  let lastMessage = "";
  document.querySelectorAll(".docstring pre").forEach(pre => {
    if (pre.closest(".fv-widget")) return;
    const wrap = document.createElement("div");
    wrap.className = "codeblock";
    pre.replaceWith(wrap);
    wrap.append(pre);
    const holder = document.createElement("span");
    holder.className = "copy";
    holder.innerHTML = `<button type="button" class="copy-button" aria-label="Copy the code">${ICON}</button><span class="copy-tip" role="status"></span>`;
    wrap.prepend(holder);
    const button = holder.querySelector("button"), tip = holder.querySelector(".copy-tip");
    button.addEventListener("click", async () => {
      const code = pre.querySelector("code") || pre;
      let ok = true;
      try { await navigator.clipboard.writeText(code.innerText.replace(/\n$/, "")); } catch { ok = false; }
      clearTimeout(timers.get(button));
      button.removeAttribute("data-copied");
      button.removeAttribute("data-failed");
      tip.textContent = "";
      void button.offsetWidth;
      button.setAttribute(ok ? "data-copied" : "data-failed", "");
      let message = ok ? "Copied" : "Copy failed";
      if (message === lastMessage) message += " ";   // a repeated message is announced again
      lastMessage = message;
      requestAnimationFrame(() => { tip.textContent = message; });
      timers.set(button, setTimeout(() => {
        button.removeAttribute("data-copied");
        button.removeAttribute("data-failed");
        tip.textContent = "";
      }, 2000));
    });
  });

  // ---- "On this page": mark the section being read.
  const toc = document.querySelector("aside.toc");
  const links = toc ? [...toc.querySelectorAll('a[href^="#"]')] : [];
  if (links.length) {
    const byId = new Map(links.map(a => [decodeURIComponent(a.hash.slice(1)), a]));
    const heads = [...byId.keys()].map(id => document.getElementById(id)).filter(Boolean);
    let last = null, frame = 0;
    const update = () => {
      frame = 0;
      const line = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 80;
      let cur = null;
      for (const h of heads) { if (h.getBoundingClientRect().top <= line + 8) cur = h; else break; }
      const a = cur ? byId.get(cur.id) : null;
      if (a === last) return;
      if (last) last.removeAttribute("aria-current");
      if (a) {
        a.setAttribute("aria-current", "true");
        const top = a.offsetTop;
        if (top < toc.scrollTop + 40 || top > toc.scrollTop + toc.clientHeight - 60) toc.scrollTop = top - toc.clientHeight / 3;
      }
      last = a;
    };
    addEventListener("scroll", () => { if (!frame) frame = requestAnimationFrame(update); }, { passive: true });
    update();
  }
})();
