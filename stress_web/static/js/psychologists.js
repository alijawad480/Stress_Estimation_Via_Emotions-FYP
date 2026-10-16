// psychologists.js
let allDoctors = [];
let selectedPsych = null;
let selectedDate = null;
let selectedTime = null;

document.addEventListener('DOMContentLoaded', loadDoctors);

async function loadDoctors() {
  const r    = await fetch('/api/psychologists');
  const data = await r.json();
  allDoctors = data.psychologists;
  renderDoctors(allDoctors);
}

function filterDoctors() {
  const city   = document.getElementById('filter-city').value;
  const fee    = parseInt(document.getElementById('filter-fee').value) || null;
  const online = document.getElementById('filter-online').value === 'true';
  const sort   = document.getElementById('filter-sort').value;

  let list = [...allDoctors];
  if (city)   list = list.filter(d => d.city === city);
  if (fee)    list = list.filter(d => d.fee_pkr <= fee);
  if (online) list = list.filter(d => d.online_available);

  if (sort === 'fee_asc')    list.sort((a,b) => a.fee_pkr - b.fee_pkr);
  else if (sort === 'experience') list.sort((a,b) => b.experience_years - a.experience_years);
  else list.sort((a,b) => b.rating - a.rating);

  renderDoctors(list);
}

function renderDoctors(list) {
  const grid = document.getElementById('doctors-grid');
  const count= document.getElementById('results-count');
  if (count) count.textContent = `${list.length} doctor${list.length!==1?'s':''} found`;

  if (!list.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon">🔍</div>
      <div>No doctors found matching your filters.</div></div>`;
    return;
  }

  grid.innerHTML = list.map(d => `
    <div class="doctor-card">
      <div class="doctor-top">
        <div class="doctor-avatar">${d.image_placeholder}</div>
        <div>
          <div class="doctor-name">${d.name}</div>
          <div class="doctor-spec">${d.specialization}</div>
          <div class="doctor-rating">★ ${d.rating} <span>(${d.reviews} reviews)</span></div>
        </div>
      </div>
      <div class="doctor-details">
        <div class="detail-item">📍 <strong>${d.city}</strong>, ${d.area}</div>
        <div class="detail-item">💰 <strong>${fmtFee(d.fee_pkr)}</strong></div>
        <div class="detail-item">🎓 ${d.experience_years} yrs exp</div>
        <div class="detail-item">⏰ ${d.timings}</div>
      </div>
      <div class="doctor-tags">
        ${d.languages.map(l => `<span class="doctor-tag">${l}</span>`).join('')}
        ${d.online_available ? '<span class="doctor-tag online-badge">Online</span>' : ''}
        ${d.availability.map(a => `<span class="doctor-tag">${a}</span>`).join('')}
      </div>
      <div class="doctor-actions">
        <button class="btn-outline" onclick="window.location='tel:${d.phone}'">📞 Call</button>
        <button class="btn-primary" onclick="openBooking('${d.id}')">Book Appointment</button>
      </div>
    </div>
  `).join('');
}

async function openBooking(psychId) {
  selectedPsych = allDoctors.find(d => d.id === psychId);
  selectedDate = selectedTime = null;
  document.getElementById('booking-modal').classList.remove('hidden');
  document.getElementById('modal-title').textContent = `Book with ${selectedPsych.name}`;
  document.getElementById('confirm-booking-btn').disabled = true;

  document.getElementById('modal-psych-info').innerHTML = `
    <div style="font-weight:600;margin-bottom:0.25rem">${selectedPsych.name}</div>
    <div style="color:var(--text2);font-size:0.82rem">${selectedPsych.specialization}</div>
    <div style="margin-top:0.5rem;display:flex;gap:1rem">
      <span>⭐ ${selectedPsych.rating}</span>
      <span>💰 ${fmtFee(selectedPsych.fee_pkr)}</span>
      <span>📍 ${selectedPsych.clinic}</span>
    </div>
  `;

  const sl = document.getElementById('slot-list');
  sl.innerHTML = '<div class="loading-spinner"></div>';
  const r    = await fetch(`/api/psychologists/${psychId}/slots`);
  const data = await r.json();

  sl.innerHTML = data.slots.map(slot => `
    <div class="slot-date">
      <div class="slot-date-label">${slot.date_display}</div>
      <div class="slot-times">
        ${slot.times.map(t => `
          <div class="slot-time" onclick="selectSlot('${slot.date}','${t}',this)">${t}</div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

function selectSlot(date, time, el) {
  document.querySelectorAll('.slot-time').forEach(s => s.classList.remove('selected'));
  el.classList.add('selected');
  selectedDate = date;
  selectedTime = time;
  document.getElementById('confirm-booking-btn').disabled = false;
}

function closeModal() {
  document.getElementById('booking-modal').classList.add('hidden');
}

async function confirmBooking() {
  if (!selectedPsych || !selectedDate || !selectedTime) return;
  const notes = document.getElementById('booking-notes').value;
  const btn   = document.getElementById('confirm-booking-btn');
  btn.disabled = true; btn.textContent = 'Confirming...';

  const r    = await fetch('/api/book', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      psychologist_id: selectedPsych.id,
      date: selectedDate, time: selectedTime, notes
    })
  });
  const data = await r.json();

  if (data.success) {
    closeModal();
    alert(`✅ ${data.message}`);
    window.location = '/dashboard';
  } else {
    alert('Booking failed. Please try again.');
    btn.disabled = false; btn.textContent = 'Confirm Appointment';
  }
}

// Click outside modal to close
document.addEventListener('click', e => {
  if (e.target.classList.contains('modal-overlay')) closeModal();
});
