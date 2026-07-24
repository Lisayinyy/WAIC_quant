/* ============================================================
   factor · ic · study — interactions v2
   - Render factor table from results.json
   - Scroll-spy for the side minimap (with numbered tags)
   - Subtle reveal-on-scroll
   ============================================================ */

(async function () {
  'use strict';

  // ---------- 1. Render factor table from JSON ----------
  try {
    const res = await fetch('data/results.json');
    if (!res.ok) throw new Error('results.json not available');
    const data = await res.json();

    const tbody = document.getElementById('factor-table-body');
    if (tbody) {
      // Compute survivors from the data itself: a survivor is a factor whose
      // partial IC (after controlling for theme / industry / size / beta /
      // liquidity / momentum) is still meaningfully large AND statistically
      // significant. This keeps the highlight honest: if the pipeline ever
      // produces different survivors, the table reflects it.
      const isSurvivor = (f) => {
        const partial = parseFloat(f.partial_ic_mean);
        const t = parseFloat(f.partial_t_stat);
        const ic = parseFloat(f.ic_mean);
        if (!isFinite(partial) || !isFinite(t) || !isFinite(ic)) return false;
        return Math.abs(ic) > 0.05 && Math.abs(partial) > 0.02 && Math.abs(t) > 2.5;
      };
      const survivors = new Set(data.factors.filter(isSurvivor).map(f => f.factor_name));

      const factors = data.factors;
      factors.sort((a, b) => {
        const aSurv = survivors.has(a.factor_name) ? 0 : 1;
        const bSurv = survivors.has(b.factor_name) ? 0 : 1;
        if (aSurv !== bSurv) return aSurv - bSurv;
        return Math.abs(b.ic_mean) - Math.abs(a.ic_mean);
      });

      const fmtIC = (v) => {
        if (v === '' || v === null || v === undefined || (typeof v === 'number' && isNaN(v))) return '—';
        const n = parseFloat(v);
        if (isNaN(n)) return '—';
        return (n >= 0 ? '+' : '') + n.toFixed(4);
      };
      const fmtP = (v) => {
        if (v === '' || v === null || v === undefined || (typeof v === 'number' && isNaN(v))) return '—';
        const n = parseFloat(v);
        if (isNaN(n)) return '—';
        if (n < 0.001) return '<0.001';
        return n.toFixed(3);
      };
      const cls = (v) => {
        const n = parseFloat(v);
        if (isNaN(n) || n === 0) return '';
        return n > 0 ? 'positive' : 'negative';
      };

      tbody.innerHTML = factors.map(f => {
        const isSurv = survivors.has(f.factor_name);
        return `
          <tr class="${isSurv ? 'survivor' : ''}">
            <td><code>${f.factor_name}</code></td>
            <td>${f.horizon}</td>
            <td class="num ${cls(f.ic_mean)}">${fmtIC(f.ic_mean)}</td>
            <td class="num">${fmtIC(f.rank_ic_mean)}</td>
            <td class="num">${parseFloat(f.icir).toFixed(3)}</td>
            <td class="num">${parseFloat(f.t_stat).toFixed(2)}</td>
            <td class="num">${fmtP(f.p_value)}</td>
            <td class="num ${cls(f.partial_ic_mean)}">${fmtIC(f.partial_ic_mean)}</td>
            <td class="num">${fmtP(f.partial_p_value)}</td>
          </tr>
        `;
      }).join('');
    }
  } catch (err) {
    console.warn('Failed to load results.json:', err);
  }

  // ---------- 2. Scroll-spy for minimap ----------
  const mmRows = Array.from(document.querySelectorAll('.mm-row'));
  const targets = mmRows
    .map(row => {
      const id = row.getAttribute('data-target');
      const el = id ? document.getElementById(id) : null;
      return el ? { row, el } : null;
    })
    .filter(Boolean);

  // Click-to-scroll
  mmRows.forEach(row => {
    const id = row.getAttribute('data-target');
    if (id) {
      row.addEventListener('click', () => {
        const el = document.getElementById(id);
        if (el) {
          const top = el.getBoundingClientRect().top + window.scrollY - 30;
          window.scrollTo({ top, behavior: 'smooth' });
        }
      });
    }
  });

  // Active row + indicator
  const indicator = document.querySelector('.minimap-indicator');
  const setActive = () => {
    const scrollY = window.scrollY;
    const viewportMid = scrollY + window.innerHeight * 0.4;
    let activeIdx = 0;
    for (let i = 0; i < targets.length; i++) {
      const { el } = targets[i];
      const rect = el.getBoundingClientRect();
      const top = rect.top + scrollY;
      if (top <= viewportMid) activeIdx = i;
    }
    mmRows.forEach((r, i) => r.classList.toggle('active', i === activeIdx));
    if (indicator) {
      const activeRow = mmRows[activeIdx];
      if (activeRow) {
        const rail = document.querySelector('.minimap-rail');
        const railTop = rail.getBoundingClientRect().top;
        const rowTop = activeRow.getBoundingClientRect().top;
        const offset = (rowTop - railTop) + 0.5;
        indicator.style.transform = `translateY(${offset}px) translateX(-12px)`;
        indicator.style.opacity = '1';
      }
    }
  };
  window.addEventListener('scroll', setActive, { passive: true });
  setActive();

  // ---------- 3. Reveal on scroll (very subtle) ----------
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
        }
      });
    }, { threshold: 0.05, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.essay-fig, .q-list li, .step-list li, .verdict, .stat').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(12px)';
      el.style.transition = 'opacity 0.55s cubic-bezier(0.25, 0.1, 0.25, 1), transform 0.55s cubic-bezier(0.25, 0.1, 0.25, 1)';
      io.observe(el);
    });
  }
})();
