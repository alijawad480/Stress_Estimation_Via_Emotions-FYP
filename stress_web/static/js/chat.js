// chat.js
document.addEventListener('DOMContentLoaded', () => {
  initSession('User').catch(() => {});
});

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const msg   = input.value.trim();
  if (!msg) return;

  input.value = ''; input.style.height = 'auto';
  appendMessage('user', msg);
  showTyping(true);

  try {
    const r    = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await r.json();
    showTyping(false);
    appendMessage('assistant', data.reply);

    // If response has doctor data, show cards
    if (data.data && data.data.psychologists) {
      showDoctorCards(data.data.psychologists);
    } else {
      hideDoctorCards();
    }

    // If booking options, show cards
    if (data.data && data.data.selected_psych) {
      showDoctorCards([data.data.selected_psych]);
    }
  } catch(e) {
    showTyping(false);
    appendMessage('assistant', 'Sorry, I had trouble responding. Please try again.');
    console.error(e);
  }
}

function sendQuick(msg) {
  document.getElementById('chat-input').value = msg;
  sendChatMessage();
}

function appendMessage(role, content) {
  const box = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${role === 'assistant' ? '🤝' : 'U'}</div>
    <div class="msg-bubble">${escapeHtml(content)}</div>
  `;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function showTyping(show) {
  document.getElementById('typing').classList.toggle('hidden', !show);
  document.getElementById('chat-send-btn').disabled = show;
}

function showDoctorCards(doctors) {
  const panel = document.getElementById('chat-cards');
  panel.classList.remove('hidden');
  panel.innerHTML = doctors.slice(0,3).map(d => `
    <div class="chat-doctor-card">
      <h4>${d.name}</h4>
      <p>📍 ${d.city} · ⭐ ${d.rating} · 💰 ${fmtFee(d.fee_pkr)}</p>
      <p style="font-size:0.75rem;color:var(--text2)">${d.specialization}</p>
      <button class="btn-primary" style="font-size:0.8rem;padding:0.4rem 0.8rem;margin-top:0.5rem"
        onclick="window.location='/psychologists'">Book →</button>
    </div>
  `).join('');
}

function hideDoctorCards() {
  const panel = document.getElementById('chat-cards');
  panel.classList.add('hidden');
  panel.innerHTML = '';
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/\n/g,'<br>');
}
