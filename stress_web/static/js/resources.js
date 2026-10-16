// resources.js
let resourcesData = null;

document.addEventListener('DOMContentLoaded', async () => {
  const r = await fetch('/api/resources');
  resourcesData = await r.json();
  showResTab('books', document.querySelector('.res-tab.active'));
});

function showResTab(type, btn) {
  document.querySelectorAll('.res-tab').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');
  if (!resourcesData) return;

  const el = document.getElementById('resources-content');
  const items = resourcesData[type] || [];

  if (type === 'books') {
    el.innerHTML = `<div class="resources-grid">${items.map(b => `
      <div class="resource-card">
        <div class="resource-top">
          <div class="resource-icon" style="background:${b.cover_color}20;font-size:1.6rem">📖</div>
          <div>
            <div class="resource-title">${b.title}</div>
            <div class="resource-author">${b.author}</div>
          </div>
        </div>
        <div class="resource-desc">${b.description}</div>
        <div class="resource-tags">
          ${b.tags.map(t => `<span class="resource-tag">${t}</span>`).join('')}
          <span class="resource-tag">${b.level}</span>
        </div>
        <div class="resource-footer">
          <div class="resource-meta">⭐ ${b.rating}</div>
          <a href="${b.buy_link}" target="_blank" class="resource-link">View Book →</a>
        </div>
      </div>
    `).join('')}</div>`;
  } else if (type === 'tutorials') {
    el.innerHTML = `<div class="resources-grid">${items.map(t => `
      <div class="resource-card">
        <div class="resource-top">
          <div class="resource-icon" style="background:rgba(127,119,221,0.1);font-size:1.3rem">${t.icon}</div>
          <div>
            <div class="resource-title">${t.title}</div>
            <div class="resource-author">${t.platform} · ${t.provider}</div>
          </div>
        </div>
        <div class="resource-desc">${t.description}</div>
        <div class="resource-tags">
          ${t.tags.map(tg => `<span class="resource-tag">${tg}</span>`).join('')}
          <span class="resource-tag">${t.level}</span>
          <span class="resource-tag" style="${t.free?'color:#1D9E75;border-color:rgba(29,158,117,0.3)':'color:#BA7517;border-color:rgba(186,117,23,0.3)'}">${t.free?'Free':'Paid'}</span>
        </div>
        <div class="resource-footer">
          <div class="resource-meta">⏱ ${t.duration}</div>
          <a href="${t.link}" target="_blank" class="resource-link">Start →</a>
        </div>
      </div>
    `).join('')}</div>`;
  } else {
    el.innerHTML = `<div class="resources-grid">${items.map(b => `
      <div class="resource-card">
        <div class="resource-top">
          <div class="resource-icon" style="background:rgba(29,158,117,0.1);font-size:1.3rem">${b.icon}</div>
          <div>
            <div class="resource-title">${b.title}</div>
            <div class="resource-author">${b.category} · ${b.language}</div>
          </div>
        </div>
        <div class="resource-desc">${b.description}</div>
        <div class="resource-footer" style="margin-top:auto">
          <div></div>
          <a href="${b.url}" target="_blank" class="resource-link">Read Blog →</a>
        </div>
      </div>
    `).join('')}</div>`;
  }
}
