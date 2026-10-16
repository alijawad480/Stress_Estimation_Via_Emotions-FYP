// dashboard.js
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const r    = await fetch('/api/session/info');
    const data = await r.json();
    renderDashboard(data);
  } catch(e) { console.error(e); }
});

function renderDashboard(data) {
  const history  = data.stress_history || [];
  const bookings = data.bookings       || [];

  if (!history.length) {
    document.getElementById('chart-wrap').classList.add('hidden');
    document.getElementById('chart-empty').classList.remove('hidden');
    ['avg-score','peak-score','sessions-count','dominant-emo']
      .forEach(id => setText(id, '—'));
  } else {
    const scores = history.map(h => h.stress_score);
    const avg    = (scores.reduce((a,b) => a+b, 0) / scores.length).toFixed(2);
    const peak   = Math.max(...scores).toFixed(2);
    const emoCounts = {};
    history.forEach(h => {
      if (h.dominant_emo) emoCounts[h.dominant_emo] = (emoCounts[h.dominant_emo]||0)+1;
    });
    const dominant = Object.entries(emoCounts).sort((a,b) => b[1]-a[1])[0]?.[0] || '—';

    setText('avg-score',      avg);
    setText('peak-score',     peak);
    setText('sessions-count', history.length);
    setText('dominant-emo',   cap(dominant));

    renderChart(history);
  }

  renderBookings(bookings);
}

function renderChart(history) {
  const ctx = document.getElementById('stress-chart');
  if (!ctx || !window.Chart) return;

  const labels = history.map((h, i) => {
    const d = new Date(h.timestamp);
    return d.toLocaleDateString('en-PK', {month:'short', day:'numeric'});
  });
  const scores = history.map(h => h.stress_score);
  const colors = history.map(h => {
    if (h.stress_label === 'Severe')   return '#D85A30';
    if (h.stress_label === 'Moderate') return '#BA7517';
    return '#1D9E75';
  });

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Stress Score',
        data: scores,
        borderColor: '#7F77DD',
        backgroundColor: 'rgba(127,119,221,0.1)',
        pointBackgroundColor: colors,
        pointRadius: 5,
        fill: true,
        tension: 0.4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { min:0, max:1, grid:{color:'rgba(255,255,255,0.05)'},
             ticks:{color:'#9898b0'} },
        x: { grid:{color:'rgba(255,255,255,0.05)'}, ticks:{color:'#9898b0'} }
      },
      plugins: {
        legend:{display:false},
        tooltip:{
          callbacks:{
            label: ctx => `Score: ${ctx.raw.toFixed(2)}`,
            afterLabel: ctx => {
              const h = (window._history||[])[ctx.dataIndex];
              return h ? `Label: ${h.stress_label} | Emotion: ${cap(h.dominant_emo)}` : '';
            }
          }
        }
      }
    }
  });
  window._history = history;
}

function renderBookings(bookings) {
  const el = document.getElementById('bookings-list');
  if (!el) return;
  if (!bookings.length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-icon">📅</div><div>No appointments yet.</div>
      <a href="/psychologists" class="btn-primary" style="margin-top:1rem;display:inline-block">Find a Doctor</a></div>`;
    return;
  }
  el.innerHTML = bookings.map(b => `
    <div class="booking-item">
      <div class="booking-info">
        <h4>${b.psychologist_name}</h4>
        <p>${b.appointment_date} at ${b.appointment_time}</p>
      </div>
      <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap">
        <span class="booking-status status-${b.status}">${cap(b.status)}</span>
        ${b.status === 'pending' ? `<button class="btn-outline" style="font-size:0.78rem;padding:0.3rem 0.7rem"
          onclick="cancelBooking(${b.id})">Cancel</button>` : ''}
      </div>
    </div>
  `).join('');
}

async function cancelBooking(id) {
  if (!confirm('Cancel this appointment?')) return;
  await fetch(`/api/bookings/${id}/cancel`, {method:'POST'});
  location.reload();
}
