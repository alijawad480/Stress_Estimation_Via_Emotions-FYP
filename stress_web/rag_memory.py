"""
rag_memory.py — RAG-based session memory
- Stores user conversations with embeddings
- Auto-deletes entries older than 7 days
- Uses ChromaDB for vector storage
- Falls back to SQLite-only if ChromaDB unavailable
"""

import os, json, sqlite3, hashlib, logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)
DB_PATH   = "data/memory.db"
CHROMA_DIR= "data/chroma_db"
RETENTION_DAYS = 7


# =============================================================================
# SQLite layer — always available
# =============================================================================

def _get_conn():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            user_name    TEXT,
            created_at   TEXT,
            last_active  TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            role         TEXT,
            content      TEXT,
            stress_score REAL,
            timestamp    TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE TABLE IF NOT EXISTS stress_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            stress_score REAL,
            stress_label TEXT,
            dominant_emo TEXT,
            modality     TEXT,
            timestamp    TEXT
        );
        CREATE TABLE IF NOT EXISTS bookings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            psychologist_id TEXT,
            psychologist_name TEXT,
            appointment_date TEXT,
            appointment_time TEXT,
            status          TEXT DEFAULT 'pending',
            notes           TEXT,
            created_at      TEXT
        );
    """)
    conn.commit(); conn.close()
    log.info("Database initialized")


def purge_old_sessions():
    """Delete sessions and related data older than RETENTION_DAYS."""
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
    conn = _get_conn()
    old = conn.execute(
        "SELECT session_id FROM sessions WHERE last_active < ?", (cutoff,)
    ).fetchall()
    for row in old:
        sid = row["session_id"]
        conn.execute("DELETE FROM messages       WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM stress_history WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM bookings       WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM sessions       WHERE session_id=?", (sid,))
        log.info("Purged session %s (older than %d days)", sid, RETENTION_DAYS)
    conn.commit(); conn.close()
    return len(old)


def create_session(session_id: str, user_name: str = "User"):
    conn = _get_conn()
    now  = datetime.now().isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO sessions (session_id, user_name, created_at, last_active)
        VALUES (?,?,?,?)
    """, (session_id, user_name, now, now))
    conn.commit(); conn.close()


def touch_session(session_id: str):
    conn = _get_conn()
    conn.execute("UPDATE sessions SET last_active=? WHERE session_id=?",
                 (datetime.now().isoformat(), session_id))
    conn.commit(); conn.close()


def save_message(session_id: str, role: str, content: str, stress_score: float = None):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO messages (session_id, role, content, stress_score, timestamp)
        VALUES (?,?,?,?,?)
    """, (session_id, role, content, stress_score, datetime.now().isoformat()))
    conn.commit(); conn.close()
    touch_session(session_id)


def get_recent_messages(session_id: str, limit: int = 20) -> list:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT role, content, timestamp FROM messages
        WHERE session_id = ?
        ORDER BY id DESC LIMIT ?
    """, (session_id, limit)).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"],
             "timestamp": r["timestamp"]} for r in reversed(rows)]


def save_stress_record(session_id: str, stress_score: float, stress_label: str,
                       dominant_emo: str, modality: str):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO stress_history
        (session_id, stress_score, stress_label, dominant_emo, modality, timestamp)
        VALUES (?,?,?,?,?,?)
    """, (session_id, stress_score, stress_label, dominant_emo, modality,
          datetime.now().isoformat()))
    conn.commit(); conn.close()


def get_stress_history(session_id: str, limit: int = 200) -> list:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT stress_score, stress_label, dominant_emo, modality, timestamp
        FROM stress_history WHERE session_id=?
        ORDER BY id DESC LIMIT ?
    """, (session_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def save_booking(session_id: str, psych_id: str, psych_name: str,
                 date: str, time: str, notes: str = "") -> int:
    conn = _get_conn()
    cur  = conn.execute("""
        INSERT INTO bookings
        (session_id, psychologist_id, psychologist_name,
         appointment_date, appointment_time, notes, created_at)
        VALUES (?,?,?,?,?,?,?)
    """, (session_id, psych_id, psych_name, date, time, notes,
          datetime.now().isoformat()))
    conn.commit(); bid = cur.lastrowid; conn.close()
    return bid


def get_bookings(session_id: str) -> list:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT * FROM bookings WHERE session_id=? ORDER BY id DESC
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cancel_booking(booking_id: int, session_id: str) -> bool:
    conn = _get_conn()
    conn.execute("""
        UPDATE bookings SET status='cancelled'
        WHERE id=? AND session_id=?
    """, (booking_id, session_id))
    conn.commit(); conn.close()
    return True


# =============================================================================
# RAG context builder
# =============================================================================

def build_context_summary(session_id: str) -> str:
    messages = get_recent_messages(session_id, limit=10)
    stress   = get_stress_history(session_id, limit=5)
    lines    = []
    if stress:
        latest = stress[-1]
        # safe key access — works with both old and new schema
        label  = latest.get("stress_label") or latest.get("stress_level") or "Unknown"
        score  = latest.get("stress_score", 0)
        emo    = latest.get("dominant_emo") or latest.get("dominant_emotion") or "unknown"
        lines.append(f"User's latest stress: {label} (score {score:.2f}), "
                     f"dominant emotion: {emo}.")
        if len(stress) > 1:
            avg = sum(s.get("stress_score", 0) for s in stress) / len(stress)
            lines.append(f"Average stress over last {len(stress)} sessions: {avg:.2f}.")
    if messages:
        lines.append("Recent conversation:")
        for m in messages[-4:]:
            role    = "User" if m["role"] == "user" else "Assistant"
            content = m["content"][:120] + "..." if len(m["content"]) > 120 else m["content"]
            lines.append(f"  {role}: {content}")
    return "\n".join(lines) if lines else "No previous context."
