import sqlite3
import random
import hashlib
import json
from typing import Optional

# ─── DATABASE SETUP ─────────────────────────────────────────────────────────
# Initialize (once at startup) a simple SQLite database
DB_PATH = "data/simulations.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        seed TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
)
cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS global_sims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
)
conn.commit()

# ─── HELPER: SEED GENERATION ─────────────────────────────────────────────────
import secrets

def make_seed(text: Optional[str] = None, length: int = 16) -> str:
    if text:
        # deterministic
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return h[:length]
    else:
        # random
        return secrets.token_hex(max(1, length//2))

def simulations_json_lookup(sim_id: int) -> str:
    """Fetch the serialized result_json for a given simulation ID."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT result_json FROM simulations WHERE id = ?",
        (sim_id,)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else '[]'

def fetch_user_sims(user_id: str, limit: int = 10):
    """Return up to `limit` most recent (id, condition, seed, created_at)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, condition, seed, created_at "
        "FROM simulations "
        "WHERE user_id = ? "
        "ORDER BY created_at DESC "
        "LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def log_global_sim(user_id: str, condition: str) -> None:
    """Record a simulation run for global statistics."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO global_sims (user_id, condition) VALUES (?, ?)",
        (user_id, condition),
    )
    conn.commit()
    conn.close()
