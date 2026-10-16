// app.js — Shared utilities and analysis functions

const API = '';  // same origin

// ── Session ────────────────────────────────────────────────────────────────
async function initSession(name = 'User') {
  const r = await fetch(`${API}/api/session`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({name})
  });
  return r.json();
}

// ── Tab switching ─────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tab);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === `tab-${tab}`);
  });
}

// ── Character counter ──────────────────────────────────────────────────────
const textInput = document.getElementById('text-input');
if (textInput) {
  textInput.addEventListener('input', () => {
    const n = textInput.value.length;
    const el = document.getElementById('char-count');
    if (el) el.textContent = `${n} / 500`;
  });
}

// ── Text Analysis ─────────────────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById('text-input').value.trim();
  if (!text) return alert('Please enter some text first.');

  const btn = document.getElementById('text-analyze-btn');
  setLoading(btn, true);

  try {
    const r = await fetch(`${API}/api/analyze/text`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    showResults(data);
  } catch(e) {
    alert('Analysis failed. Make sure the server is running.');
    console.error(e);
  } finally {
    setLoading(btn, false);
  }
}

// ── Audio file handling ────────────────────────────────────────────────────
let audioBlob = null;
let mediaRecorder = null;
let recordingChunks = [];
let recordTimer = null;
let recordSeconds = 0;

function handleAudioFile(input) {
  if (!input.files[0]) return;
  audioBlob = input.files[0];
  document.getElementById('audio-filename').textContent = input.files[0].name;
  document.getElementById('audio-analyze-btn').disabled = false;
}

async function toggleRecording() {
  const btn = document.getElementById('record-btn');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    clearInterval(recordTimer);
    btn.textContent = '🎙 Start Recording';
    btn.classList.remove('recording');
    document.getElementById('record-timer').classList.add('hidden');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    mediaRecorder = new MediaRecorder(stream);
    recordingChunks = [];
    mediaRecorder.ondataavailable = e => recordingChunks.push(e.data);
    mediaRecorder.onstop = () => {
      audioBlob = new Blob(recordingChunks, {type: 'audio/wav'});
      document.getElementById('audio-filename').textContent = `Recording (${recordSeconds}s)`;
      document.getElementById('audio-analyze-btn').disabled = false;
      stream.getTracks().forEach(t => t.stop());
    };
    mediaRecorder.start();
    recordSeconds = 0;
    document.getElementById('record-timer').classList.remove('hidden');
    recordTimer = setInterval(() => {
      recordSeconds++;
      const m = Math.floor(recordSeconds/60);
      const s = recordSeconds % 60;
      document.getElementById('record-timer').textContent = `${m}:${s.toString().padStart(2,'0')}`;
    }, 1000);
    btn.textContent = '⏹ Stop Recording';
    btn.classList.add('recording');
  } catch(e) {
    alert('Microphone access denied.');
  }
}

async function analyzeAudio() {
  if (!audioBlob) return;
  const btn = document.getElementById('audio-analyze-btn');
  setLoading(btn, true);
  const fd = new FormData();
  fd.append('file', audioBlob, 'audio.wav');
  try {
    const r = await fetch(`${API}/api/analyze/audio`, {method:'POST', body:fd});
    const data = await r.json();
    showResults(data);
  } catch(e) {
    alert('Analysis failed.'); console.error(e);
  } finally { setLoading(btn, false); }
}

// ── Video file handling ────────────────────────────────────────────────────
let videoFile = null;

function handleVideoFile(input) {
  if (!input.files[0]) return;
  videoFile = input.files[0];
  document.getElementById('video-filename').textContent = input.files[0].name;
  document.getElementById('video-analyze-btn').disabled = false;
}

async function analyzeVideo() {
  if (!videoFile) return;
  const btn = document.getElementById('video-analyze-btn');
  setLoading(btn, true);
  const fd = new FormData();
  fd.append('file', videoFile, videoFile.name);
  try {
    const r = await fetch(`${API}/api/analyze/video`, {method:'POST', body:fd});
    const data = await r.json();
    showResults(data);
  } catch(e) {
    alert('Analysis failed.'); console.error(e);
  } finally { setLoading(btn, false); }
}

// ── Show Results ──────────────────────────────────────────────────────────
const EMOTION_COLORS = {
  anger:'#D85A30', fear:'#993556', happy:'#1D9E75',
  sad:'#185FA5', surprise:'#BA7517', disgust:'#534AB7'
};

function showResults(data) {
  const panel = document.getElementById('result-panel');
  if (!panel) return;
  panel.classList.remove('hidden');
  panel.scrollIntoView({behavior:'smooth', block:'nearest'});

  const score = data.stress_score || 0;
  const label = data.stress_label || 'Low';

  // Gauge
  const totalLen = 173;
  const filled   = totalLen * Math.min(score, 1.0);
  const gaugePath = document.getElementById('gauge-path');
  if (gaugePath) {
    gaugePath.style.strokeDasharray = `${filled} ${totalLen}`;
    gaugePath.style.stroke = label === 'Severe' ? '#D85A30' :
                             label === 'Moderate'? '#BA7517' : '#1D9E75';
  }
  setText('gauge-value', score.toFixed(2));
  setText('gauge-label', label);

  // Label badge
  const lb = document.getElementById('result-label');
  if (lb) {
    lb.textContent = label;
    lb.className = `badge badge-${label.toLowerCase()}`;
  }
  setText('result-dominant', `Dominant: ${cap(data.dominant_emotion || '—')}`);

  // Audio note
  const note = document.getElementById('result-note');
  if (note && data.audio_note) {
    note.textContent = data.audio_note; note.classList.remove('hidden');
  } else if (note) { note.classList.add('hidden'); }

  // Demo note
  if (data.ml_model === 'demo') {
    const n = document.getElementById('result-note');
    if (n) { n.textContent = data.note; n.classList.remove('hidden'); }
  }

  // Emotion bars
  const bars = document.getElementById('emotion-bars');
  if (bars && data.emotions) {
    const maxVal = Math.max(...Object.values(data.emotions), 0.1);
    bars.innerHTML = Object.entries(data.emotions).map(([name, val]) => `
      <div class="emo-row">
        <div class="emo-name">${cap(name)}</div>
        <div class="emo-bar-wrap">
          <div class="emo-bar" style="width:${(val/3*100).toFixed(1)}%;background:${EMOTION_COLORS[name]||'#7F77DD'}"></div>
        </div>
        <div class="emo-val">${val.toFixed(2)}</div>
      </div>
    `).join('');
  }

  // Update hero orb if on home page
  // ── Update hero orb ─────────────────────────────────────
  const orbScoreEl = document.querySelector('.orb-score');
  const orbLabelEl = document.querySelector('.orb-label');
  const orbCenter  = document.querySelector('.orb-center');

  if (orbScoreEl) orbScoreEl.textContent = score.toFixed(2);
  if (orbLabelEl) orbLabelEl.textContent = label;
  if (orbCenter) {
    orbCenter.style.borderColor =
      label === 'Severe'   ? '#D85A30' :
      label === 'Moderate' ? '#BA7517' : '#1D9E75';
    orbCenter.style.transition = 'border-color 0.5s';
  }

  // Also show warning if no human content detected
  const noteEl = document.getElementById('result-note');
  if (noteEl && data.warning) {
    noteEl.textContent = data.warning;
    noteEl.classList.remove('hidden');
  }

  // Update pills
  if (data.emotions) {
    document.querySelectorAll('.pill').forEach(p => {
      const emo = p.dataset.emotion;
      const val = data.emotions[emo] || 0;
      p.classList.toggle('active', val > 0.5);
    });
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setLoading(btn, loading) {
  if (!btn) return;
  const txt = btn.querySelector('.btn-text');
  const ldr = btn.querySelector('.btn-loader');
  if (txt && ldr) {
    txt.classList.toggle('hidden',  loading);
    ldr.classList.toggle('hidden', !loading);
  } else {
    btn.disabled = loading;
    if (loading) btn.dataset.orig = btn.textContent;
    else if (btn.dataset.orig) btn.textContent = btn.dataset.orig;
  }
  btn.disabled = loading;
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function cap(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
}

function stars(rating) {
  const full = Math.floor(rating);
  const half = rating % 1 >= 0.5 ? 1 : 0;
  return '★'.repeat(full) + (half ? '½' : '') + '☆'.repeat(5-full-half);
}

function fmtFee(pkr) {
  return 'PKR ' + pkr.toLocaleString();
}

// ── Init session on load ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSession('User').catch(() => {});
});
