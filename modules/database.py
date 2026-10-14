import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import config

def get_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with Row factory."""
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processed'
            )
        """)
        
        # Verification Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                document_name TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                model_used TEXT NOT NULL,
                total_claims INTEGER NOT NULL,
                supported_claims INTEGER NOT NULL,
                contradicted_claims INTEGER NOT NULL,
                unsupported_claims INTEGER NOT NULL,
                overall_verdict TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE SET NULL
            )
        """)
        
        # Claims table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                nli_relation TEXT NOT NULL,
                entailment_score REAL NOT NULL,
                contradiction_score REAL NOT NULL,
                neutral_score REAL NOT NULL,
                evidence_text TEXT,
                evidence_page INTEGER,
                evidence_score REAL,
                judge_assessment TEXT,
                judge_used INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES verification_sessions(id) ON DELETE CASCADE
            )
        """)
        conn.commit()

def save_document(filename: str, page_count: int, chunk_count: int, file_hash: Optional[str] = None) -> int:
    """Saves or updates document metadata and returns document ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        if file_hash:
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute("""
                    UPDATE documents 
                    SET filename = ?, page_count = ?, chunk_count = ?, upload_time = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (filename, page_count, chunk_count, existing["id"]))
                conn.commit()
                return existing["id"]
        
        cursor.execute("""
            INSERT INTO documents (filename, file_hash, page_count, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (filename, file_hash, page_count, chunk_count))
        conn.commit()
        return cursor.lastrowid

def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieves all registered documents."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY upload_time DESC")
        return [dict(row) for row in cursor.fetchall()]

def save_verification_session(
    document_name: str,
    question: str,
    answer: str,
    model_used: str,
    claims: List[Dict[str, Any]],
    doc_id: Optional[int] = None
) -> int:
    """Saves a complete verification session and all associated claims atomically."""
    init_db()
    
    total_claims = len(claims)
    supported_claims = sum(1 for c in claims if c.get("verdict") == config.VERDICT_SUPPORTED)
    contradicted_claims = sum(1 for c in claims if c.get("verdict") == config.VERDICT_CONTRADICTED)
    unsupported_claims = sum(1 for c in claims if c.get("verdict") == config.VERDICT_UNSUPPORTED)
    
    # Calculate overall answer verdict
    if contradicted_claims > 0:
        overall_verdict = "Contains Contradictions"
    elif unsupported_claims > 0:
        overall_verdict = "Partially Supported / Unverified"
    elif total_claims > 0 and supported_claims == total_claims:
        overall_verdict = "Fully Supported"
    else:
        overall_verdict = "Inconclusive"
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO verification_sessions (
                doc_id, document_name, question, answer, model_used,
                total_claims, supported_claims, contradicted_claims, unsupported_claims,
                overall_verdict
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, document_name, question, answer, model_used,
            total_claims, supported_claims, contradicted_claims, unsupported_claims,
            overall_verdict
        ))
        session_id = cursor.lastrowid
        
        for c in claims:
            cursor.execute("""
                INSERT INTO claims (
                    session_id, claim_text, verdict, nli_relation,
                    entailment_score, contradiction_score, neutral_score,
                    evidence_text, evidence_page, evidence_score,
                    judge_assessment, judge_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                c.get("claim_text", ""),
                c.get("verdict", config.VERDICT_UNSUPPORTED),
                c.get("nli_relation", "Neutral"),
                float(c.get("entailment_score", 0.0)),
                float(c.get("contradiction_score", 0.0)),
                float(c.get("neutral_score", 0.0)),
                c.get("evidence_text", ""),
                c.get("evidence_page"),
                float(c.get("evidence_score", 0.0)) if c.get("evidence_score") is not None else None,
                c.get("judge_assessment", ""),
                1 if c.get("judge_used") else 0
            ))
            
        conn.commit()
        return session_id

def get_all_sessions() -> List[Dict[str, Any]]:
    """Retrieves all previous verification sessions with summary counts."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM verification_sessions 
            ORDER BY timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

def get_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single session along with all its detailed claims."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verification_sessions WHERE id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            return None
        
        session = dict(session_row)
        cursor.execute("SELECT * FROM claims WHERE session_id = ? ORDER BY id ASC", (session_id,))
        session["claims"] = [dict(r) for r in cursor.fetchall()]
        return session

def delete_session(session_id: int) -> bool:
    """Deletes a verification session and associated claims."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM claims WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM verification_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

def clear_all_history() -> None:
    """Clears all stored verification sessions and claims."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM claims")
        cursor.execute("DELETE FROM verification_sessions")
        conn.commit()

def get_system_stats() -> Dict[str, int]:
    """Returns real metrics from the database without hardcoded mock statistics."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM verification_sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM claims")
        total_claims = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM claims 
            WHERE verdict IN (?, ?)
        """, (config.VERDICT_CONTRADICTED, config.VERDICT_UNSUPPORTED))
        flagged_claims = cursor.fetchone()[0]
        
        return {
            "documents_processed": total_docs,
            "verification_sessions": total_sessions,
            "claims_checked": total_claims,
            "claims_flagged": flagged_claims
        }
