"""
KingsGuard Dashboard — Flask Backend
Provides SSE-based real-time pipeline streaming, dataset analysis, and L1 benchmarking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-load heavyweight libraries before torch to prevent Windows Kernel crashes
import sklearn
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, jsonify, request, Response
import threading
import queue
import json
import time
import sqlite3
import uuid
import traceback
import random
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR   = Path(__file__).parent
# Flexible path searching for datasets and storage
DATA_DIR   = next((p for p in [BASE_DIR / "data" / "datasets" / "injecagent_data", BASE_DIR / "injecagent_data"] if p.exists()), BASE_DIR / "data" / "datasets" / "injecagent_data")
DB_PATH    = next((p for p in [BASE_DIR / "data" / "storage" / "final_causal_model.db", BASE_DIR / "final_causal_model.db"] if p.exists()), BASE_DIR / "data" / "storage" / "final_causal_model.db")
TRUST_DB   = next((p for p in [BASE_DIR / "data" / "storage" / "trust.db", BASE_DIR / "trust.db"] if p.exists()), BASE_DIR / "data" / "storage" / "trust.db")

# ─────────────────────────────────────────────────────────────
# Global state & DB Init
# ─────────────────────────────────────────────────────────────
HISTORY_DB = BASE_DIR / "data" / "storage" / "history.db"

def init_history_db():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            prompt TEXT,
            agent_id TEXT,
            verdict TEXT,
            avg_risk REAL,
            total_ms INTEGER,
            layers_json TEXT
        )
    """)
    conn.commit()
    conn.close()

init_history_db()

def save_history(job_id, prompt, agent_id, verdict, avg_risk, total_ms, layer_results):
    try:
        conn = sqlite3.connect(HISTORY_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO history (id, timestamp, prompt, agent_id, verdict, avg_risk, total_ms, layers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, datetime.now().isoformat(), prompt[:200], agent_id, verdict, round(avg_risk, 4), total_ms, json.dumps(layer_results)))
        conn.commit()
        conn.close()
        print(f"[DB] Saved history for job {job_id}")
    except Exception as e:
        print(f"[DB] History save failed for job {job_id}: {e}")

_jobs: dict = {}          # job_id → {"queue": Queue, "status": str}
_jobs_lock = threading.Lock()

_preload = {
    "l1":  {"status": "pending", "time_ms": 0, "error": None},
    "ppl": {"status": "pending", "time_ms": 0, "error": None},
    "l2":  {"status": "pending", "time_ms": 0, "error": None},
}
_preload_lock = threading.Lock()
_preload_started = False

# ─────────────────────────────────────────────────────────────
# Model Pre-loading
# ─────────────────────────────────────────────────────────────
def _load_model(key: str, loader_fn):
    with _preload_lock:
        _preload[key]["status"] = "loading"
    try:
        t0 = time.time()
        loader_fn()
        ms = round((time.time() - t0) * 1000)
        with _preload_lock:
            _preload[key]["status"] = "ready"
            _preload[key]["time_ms"] = ms
        print(f"[PRELOAD] {key.upper()} ready in {ms} ms")
    except Exception as e:
        with _preload_lock:
            _preload[key]["status"] = "error"
            _preload[key]["error"] = str(e)[:120]
        print(f"[PRELOAD] {key.upper()} failed: {e}")


def _preload_all():
    from tools import get_l1_model, get_ppl_model, get_l2_model
    _load_model("l1",  get_l1_model)
    _load_model("ppl", get_ppl_model)
    _load_model("l2",  get_l2_model)


def start_preloading():
    global _preload_started
    if not _preload_started:
        _preload_started = True
        threading.Thread(target=_preload_all, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# Dataset Analysis
# ─────────────────────────────────────────────────────────────
def analyze_dataset_generic() -> list:
    """Scans for all datasets and extracts metadata and samples."""
    datasets = []
    
    # Search paths for datasets
    search_dirs = [
        BASE_DIR / "data" / "datasets",
        BASE_DIR / "data" / "datasets" / "injecagent_data",
        BASE_DIR / "data" / "datasets" / "Prompt-injection-dataset"
    ]
    
    seen_files = set()
    
    for ddir in search_dirs:
        if not ddir.exists():
            continue
        # Search recursively for parquet and others
        for fpath in ddir.rglob("*"):
            if fpath.suffix.lower() not in [".csv", ".json", ".jsonl", ".parquet"]:
                continue
            if fpath.name in seen_files:
                continue
            
            seen_files.add(fpath.name)
            
            ds_info = {
                "name": fpath.name,
                "path": str(fpath),
                "type": fpath.suffix[1:].upper(),
                "size_kb": round(fpath.stat().st_size / 1024),
                "records": 0,
                "samples": []
            }
            
            try:
                df = None
                if fpath.suffix == ".csv":
                    df = pd.read_csv(fpath)
                elif fpath.suffix == ".parquet":
                    df = pd.read_parquet(fpath)
                
                if df is not None:
                    ds_info["records"] = len(df)
                    # Convert to dict and handle numpy types for JSON
                    samples = df.head(5).to_dict("records")
                    # Recursive conversion of numpy types
                    def fix_types(obj):
                        if isinstance(obj, dict):
                            return {k: fix_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [fix_types(i) for i in obj]
                        elif hasattr(obj, "item"): # numpy types
                            return obj.item()
                        elif pd.isna(obj):
                            return None
                        return obj
                    ds_info["samples"] = fix_types(samples)

                elif fpath.suffix == ".json":
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        ds_info["records"] = len(data)
                        ds_info["samples"] = data[:5]
                    else:
                        ds_info["records"] = 1
                        ds_info["samples"] = [data]
                elif fpath.suffix == ".jsonl":
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = [json.loads(line) for line in f if line.strip()]
                    ds_info["records"] = len(lines)
                    ds_info["samples"] = lines[:5]
            except Exception as e:
                ds_info["error"] = str(e)
                
            datasets.append(ds_info)
            
    return datasets


# ─────────────────────────────────────────────────────────────
# L1 Benchmark Runner
# ─────────────────────────────────────────────────────────────
@app.route("/api/datasets/list")
def list_datasets():
    """Returns list of all discovered datasets (names)."""
    return jsonify([d["name"] for d in analyze_dataset_generic()])

@app.route("/api/dataset/benchmark", methods=["POST"])
def run_benchmark_api():
    data = request.json
    filename = data.get("dataset", "MPDD_test.csv")
    n_samples = int(data.get("samples", 30))

    # Search recursively for the file
    search_dirs = [
        BASE_DIR / "data" / "datasets",
        BASE_DIR / "data" / "datasets" / "injecagent_data",
        BASE_DIR / "data" / "datasets" / "Prompt-injection-dataset"
    ]
    path = None
    for ddir in search_dirs:
        if not ddir.exists(): continue
        for p in ddir.rglob(filename):
            path = p
            break
        if path: break
    
    if not path:
        return jsonify({"error": f"Dataset {filename} not found."}), 404

    # Sampling logic
    records = []
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            # Find columns
            txt_col = next((c for c in df.columns if "text" in c.lower() or "instruction" in c.lower()), df.columns[0])
            lbl_col = next((c for c in df.columns if "label" in c.lower()), None)
            
            if lbl_col:
                mal = df[df[lbl_col] == 1]
                ben = df[df[lbl_col] == 0]
                s_mal = mal.sample(min(n_samples//2, len(mal)))
                s_ben = ben.sample(min(n_samples//2, len(ben)))
                for _, r in s_mal.iterrows(): records.append((str(r[txt_col]), "malicious"))
                for _, r in s_ben.iterrows(): records.append((str(r[txt_col]), "benign"))
            else:
                s = df.sample(min(n_samples, len(df)))
                for _, r in s.iterrows(): records.append((str(r[txt_col]), "unknown"))
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            txt_col = next((c for c in df.columns if "text" in c.lower() or "instruction" in c.lower()), df.columns[0])
            lbl_col = next((c for c in df.columns if "label" in c.lower()), None)
            if lbl_col:
                mal = df[df[lbl_col] == 1]
                ben = df[df[lbl_col] == 0]
                s_mal = mal.sample(min(n_samples//2, len(mal)))
                s_ben = ben.sample(min(n_samples//2, len(ben)))
                for _, r in s_mal.iterrows(): records.append((str(r[txt_col]), "malicious"))
                for _, r in s_ben.iterrows(): records.append((str(r[txt_col]), "benign"))
            else:
                s = df.sample(min(n_samples, len(df)))
                for _, r in s.iterrows(): records.append((str(r[txt_col]), "unknown"))
        else:
            # JSON/JSONL
            raw_data = []
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = [json.loads(l) for l in f if l.strip()]
            
            if not isinstance(raw_data, list): raw_data = [raw_data]
            s = random.sample(raw_data, min(n_samples, len(raw_data)))
            for rec in s:
                txt = rec.get("text") or rec.get("User Instruction") or rec.get("Attacker Instruction") or str(rec)
                lbl = "malicious" if rec.get("label") == 1 or rec.get("Modifed") == 1 else "benign"
                records.append((txt, lbl))
    except Exception as e:
        return jsonify({"error": f"Error loading/sampling {filename}: {str(e)}"}), 500

    from tools import (
        KingsGuardL1Tool, PerplexityCalcTool, KingsGuardL2Tool, 
        KingsGuardL3Tool, SecurityException
    )
    l1_tool = KingsGuardL1Tool()
    ppl_tool = PerplexityCalcTool()
    l2_tool = KingsGuardL2Tool()
    l3_tool = KingsGuardL3Tool()

    tp = fp = tn = fn = 0
    rows = []
    times = []

    for text, true_label in records:
        t0 = time.time()
        layers = {}
        final_pred = "benign"
        
        try:
            # L1
            l1_score = 0.0
            try:
                res1 = json.loads(l1_tool._run(text=text))
                l1_score = res1["security_metadata"]["threat_score"]
                layers["L1"] = {"status": "PASS", "score": l1_score}
            except SecurityException:
                layers["L1"] = {"status": "FAIL", "score": 1.0}
                final_pred = "malicious"

            # PPL
            try:
                ppl_tool._run(prompt=text)
                layers["PPL"] = {"status": "PASS", "score": 0.0}
            except SecurityException:
                layers["PPL"] = {"status": "FAIL", "score": 1000.0}
                final_pred = "malicious"

            # L2
            res2 = json.loads(l2_tool._run(agent_action_text=text))
            l2_mse = res2["score"]
            layers["L2"] = {"status": res2["status"], "score": l2_mse}

            # L3
            res3 = json.loads(l3_tool._run(user_query=text, agent_action=text, l1_score=l1_score, l2_mse=l2_mse))
            layers["L3"] = {"status": res3["status"], "score": res3.get("avg_risk", 0.0)}
            if res3["status"] == "QUARANTINE":
                final_pred = "malicious"
        except Exception:
            final_pred = "error"

        ms = round((time.time() - t0) * 1000)
        times.append(ms)
        correct = (final_pred == true_label)

        if final_pred == "malicious" and true_label == "malicious": tp += 1
        elif final_pred == "malicious" and true_label == "benign":  fp += 1
        elif final_pred == "benign"    and true_label == "benign":  tn += 1
        elif final_pred == "benign"    and true_label == "malicious": fn += 1

        rows.append({
            "text": text[:100] + "...", "true_label": true_label,
            "pred_label": final_pred, "layers": layers,
            "correct": correct, "ms": ms
        })

    total = tp + fp + tn + fn
    accuracy = round((tp + tn) / max(total, 1), 4)
    return jsonify({
        "metrics": {"accuracy": accuracy, "avg_ms": round(sum(times)/max(len(times),1))},
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "rows": rows
    })


# ─────────────────────────────────────────────────────────────
# Pipeline Runner  (runs in background thread)
# ─────────────────────────────────────────────────────────────
def _push(q, event_type, data):
    q.put(json.dumps({"type": event_type, "data": data, "ts": time.time()}))


def run_pipeline(job_id: str, prompt: str, agent_id: str):
    q = _jobs[job_id]["queue"]
    t_total = time.time()
    layer_results = {}

    _push(q, "pipeline_start", {
        "prompt": prompt, "agent_id": agent_id,
        "timestamp": datetime.now().isoformat(),
    })

    def run_layer(layer_id, name, role, fn):
        _push(q, "layer_start", {"layer": layer_id, "name": name, "role": role})
        t0 = time.time()
        try:
            result = fn()
            ms = round((time.time() - t0) * 1000)
            layer_results[layer_id] = {"status": "ok", "result": result, "ms": ms}
            _push(q, "layer_complete", {"layer": layer_id, "ms": ms, **result})
            return result
        except Exception as e:
            ms = round((time.time() - t0) * 1000)
            err = {"status": "ERROR", "error": str(e), "ms": ms}
            layer_results[layer_id] = err
            _push(q, "layer_error", {"layer": layer_id, "ms": ms, "error": str(e)})
            return None

    try:
        import torch
        import torch.nn.functional as F
        from tools import (
            KingsGuardL1Tool, PerplexityCalcTool,
            KingsGuardL2Tool, KingsGuardL3Tool,
            TrustScoreTool,
            SecurityException, KingsGuardSecurityBreach,
        )

        # ── L4 PRE-CHECK ──────────────────────────────────────
        _push(q, "layer_start", {"layer": "L4_pre", "name": "Watchman Pre-Check", "role": "Behavioral Monitor"})
        t0 = time.time()
        try:
            res = json.loads(TrustScoreTool()._run(agent_id=agent_id, score_update=0.0, check_only=True))
            ms  = round((time.time() - t0) * 1000)
            layer_results["L4_pre"] = {"status": "NORMAL", "result": res, "ms": ms}
            _push(q, "layer_complete", {"layer": "L4_pre", "status": "NORMAL", "ms": ms, "result": res})
        except KingsGuardSecurityBreach as e:
            ms  = round((time.time() - t0) * 1000)
            _push(q, "layer_error",    {"layer": "L4_pre", "status": "REVOKED", "ms": ms, "error": str(e)})
            final = {
                "verdict": "BLOCKED", "reason": "Agent trust REVOKED",
                "layers": layer_results, "total_ms": round((time.time() - t_total) * 1000),
            }
            _push(q, "pipeline_complete", final)
            save_history(job_id, prompt, agent_id, "BLOCKED", 1.0, final["total_ms"], layer_results)
            _jobs[job_id]["status"] = "complete"
            _push(q, "__done__", {})
            return

        # ── L1 SEMANTIC SCREENER ──────────────────────────────
        _push(q, "layer_start", {"layer": "L1", "name": "Intent Screener", "role": "Semantic Intent Classifier"})
        t0 = time.time()
        l1_confidence = 0.1
        try:
            raw = KingsGuardL1Tool()._run(text=prompt)
            res = json.loads(raw)
            ms  = round((time.time() - t0) * 1000)
            l1_confidence = res.get("security_metadata", {}).get("confidence", 0.1)
            layer_results["L1"] = {"status": res.get("status", "UNKNOWN"), "result": res, "ms": ms}
            _push(q, "layer_complete", {"layer": "L1", "ms": ms, **layer_results["L1"]})
        except SecurityException as e:
            ms = round((time.time() - t0) * 1000)
            _push(q, "layer_error", {"layer": "L1", "status": "THREAT_DETECTED", "ms": ms, "error": str(e)})
            final = {
                "verdict": "BLOCKED", "reason": str(e),
                "layers": layer_results, "total_ms": round((time.time() - t_total) * 1000),
            }
            _push(q, "pipeline_complete", final)
            save_history(job_id, prompt, agent_id, "BLOCKED", 1.0, final["total_ms"], layer_results)
            _jobs[job_id]["status"] = "complete"
            _push(q, "__done__", {})
            return

        # ── L1 PERPLEXITY ─────────────────────────────────────
        _push(q, "layer_start", {"layer": "L1_ppl", "name": "Perplexity Filter", "role": "Adversarial Camouflage Detector"})
        t0 = time.time()
        ppl_score = 0.0
        try:
            raw = PerplexityCalcTool()._run(prompt=prompt)
            res = json.loads(raw)
            ms  = round((time.time() - t0) * 1000)
            ppl_score = res.get("security_metadata", {}).get("perplexity", 0.0)
            layer_results["L1_ppl"] = {"status": res.get("status", "UNKNOWN"), "result": res, "ms": ms}
            _push(q, "layer_complete", {"layer": "L1_ppl", "ms": ms, **layer_results["L1_ppl"]})
        except SecurityException as e:
            ms = round((time.time() - t0) * 1000)
            _push(q, "layer_error", {"layer": "L1_ppl", "status": "THREAT_DETECTED", "ms": ms, "error": str(e)})
            final = {
                "verdict": "BLOCKED", "reason": str(e),
                "layers": layer_results, "total_ms": round((time.time() - t_total) * 1000),
            }
            _push(q, "pipeline_complete", final)
            save_history(job_id, prompt, agent_id, "BLOCKED", 1.0, final["total_ms"], layer_results)
            _jobs[job_id]["status"] = "complete"
            _push(q, "__done__", {})
            return

        # ── L2 VAE PROFILER ───────────────────────────────────
        _push(q, "layer_start", {"layer": "L2", "name": "VAE Anomaly Profiler", "role": "Zero-Day Anomaly Detector"})
        t0 = time.time()
        l2_mse = 0.01
        try:
            raw = KingsGuardL2Tool()._run(agent_action_text=prompt)
            res = json.loads(raw)
            ms  = round((time.time() - t0) * 1000)
            l2_mse = res.get("score", 0.01)
            layer_results["L2"] = {"status": res.get("status", "UNKNOWN"), "result": res, "ms": ms}
            _push(q, "layer_complete", {"layer": "L2", "ms": ms, **layer_results["L2"]})
        except Exception as e:
            ms = round((time.time() - t0) * 1000)
            l2_mse = 0.01
            layer_results["L2"] = {"status": "ERROR", "error": str(e), "ms": ms}
            _push(q, "layer_error", {"layer": "L2", "ms": ms, "error": str(e)})

        # ── L3 CAUSAL ARBITRATOR ──────────────────────────────
        _push(q, "layer_start", {"layer": "L3", "name": "Council of Rivals", "role": "Causal Logic Gate"})
        t0 = time.time()
        avg_risk = 0.5
        try:
            raw = KingsGuardL3Tool()._run(
                user_query=prompt,
                agent_action=f"Execute: {prompt}",
                l1_score=float(l1_confidence),
                l2_mse=float(l2_mse),
            )
            res = json.loads(raw)
            ms  = round((time.time() - t0) * 1000)
            avg_risk = float(res.get("avg_risk", 0.5))
            layer_results["L3"] = {"status": res.get("status", "UNKNOWN"), "result": res, "ms": ms}
            _push(q, "layer_complete", {"layer": "L3", "ms": ms, **layer_results["L3"]})
        except Exception as e:
            ms = round((time.time() - t0) * 1000)
            layer_results["L3"] = {"status": "ERROR", "error": str(e), "ms": ms}
            _push(q, "layer_error", {"layer": "L3", "ms": ms, "error": str(e)})

        # ── L4 POST-CHECK ─────────────────────────────────────
        _push(q, "layer_start", {"layer": "L4_post", "name": "Trust Score Update", "role": "Behavioral Monitor"})
        t0 = time.time()
        try:
            res = json.loads(TrustScoreTool()._run(
                agent_id=agent_id, score_update=round(avg_risk, 4), check_only=False
            ))
            ms  = round((time.time() - t0) * 1000)
            layer_results["L4_post"] = {"status": "NORMAL", "result": res, "ms": ms}
            _push(q, "layer_complete", {"layer": "L4_post", "status": "NORMAL", "ms": ms, "result": res})
        except KingsGuardSecurityBreach as e:
            ms = round((time.time() - t0) * 1000)
            layer_results["L4_post"] = {"status": "REVOKED", "error": str(e), "ms": ms}
            _push(q, "layer_error", {"layer": "L4_post", "status": "TRUST_CLIFF", "ms": ms, "error": str(e)})

        # ── FINAL VERDICT ─────────────────────────────────────
        l3_status = layer_results.get("L3", {}).get("status", "UNKNOWN")
        l4_revoked = layer_results.get("L4_post", {}).get("status") == "REVOKED"

        if l4_revoked:
            verdict = "BLOCKED"
        elif l3_status == "QUARANTINE":
            verdict = "QUARANTINE"
        elif l3_status == "APPROVED":
            verdict = "APPROVED"
        else:
            verdict = "UNKNOWN"

        total_ms = round((time.time() - t_total) * 1000)
        final = {
            "verdict": verdict,
            "avg_risk": round(avg_risk, 4),
            "layers": layer_results,
            "total_ms": total_ms,
        }

        _push(q, "pipeline_complete", final)
        save_history(job_id, prompt, agent_id, verdict, avg_risk, total_ms, layer_results)

    except Exception as e:
        _push(q, "pipeline_error", {"error": str(e), "traceback": traceback.format_exc()})
        save_history(job_id, prompt, agent_id, "ERROR", 0.0, 0, {"error": str(e)})
        _push(q, "pipeline_error", {"error": str(e), "traceback": traceback.format_exc()})
    finally:
        _jobs[job_id]["status"] = "complete"
        _push(q, "__done__", {})


# ─────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preload/status")
def preload_status():
    with _preload_lock:
        return jsonify(_preload)


@app.route("/api/preload/trigger", methods=["POST"])
def preload_trigger():
    start_preloading()
    return jsonify({"ok": True})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    prompt   = body.get("prompt", "").strip()
    agent_id = body.get("agent_id", "default_user").strip() or "default_user"
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"queue": queue.Queue(), "status": "running"}

    threading.Thread(target=run_pipeline, args=(job_id, prompt, agent_id), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({"error": "unknown job"}), 404
    q = _jobs[job_id]["queue"]

    def generate():
        while True:
            try:
                msg = q.get(timeout=30)
                if json.loads(msg).get("type") == "__done__":
                    yield f"data: {msg}\n\n"
                    break
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/datasets/info")
def datasets_info():
    return jsonify(analyze_dataset_generic())

# Benchmarking results storage
BENCH_RESULTS_FILE = BASE_DIR / "data" / "storage" / "bench_all_results.json"
_bench_status = {"status": "idle", "progress": 0, "current_dataset": ""}

def _run_full_bench():
    global _bench_status
    datasets = analyze_dataset_generic()
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {}
    }
    
    # Pre-load results if file exists to resume or append
    if BENCH_RESULTS_FILE.exists():
        try:
            with open(BENCH_RESULTS_FILE, "r") as f:
                old = json.load(f)
                all_results["datasets"] = old.get("datasets", {})
        except: pass

    from tools import (
        KingsGuardL1Tool, PerplexityCalcTool, KingsGuardL2Tool, 
        KingsGuardL3Tool, SecurityException
    )
    l1_tool = KingsGuardL1Tool()
    ppl_tool = PerplexityCalcTool()
    l2_tool = KingsGuardL2Tool()
    l3_tool = KingsGuardL3Tool()
    
    total_ds = len(datasets)
    for i, ds in enumerate(datasets):
        name = ds["name"]
        _bench_status["current_dataset"] = name
        _bench_status["progress"] = round((i / total_ds) * 100)
        
        # Skip if already done in this run? No, user wants fresh but save whats done if crash.
        
        # Sample 30
        records = []
        try:
            if ds["type"] == "CSV":
                df = pd.read_csv(ds["path"])
                if "label" in df.columns:
                    mal = df[df["label"] == 1]
                    ben = df[df["label"] == 0]
                    s_mal = mal.sample(min(15, len(mal)))
                    s_ben = ben.sample(min(15, len(ben)))
                    for _, r in s_mal.iterrows(): records.append((r["text"], "malicious"))
                    for _, r in s_ben.iterrows(): records.append((r["text"], "benign"))
                else:
                    s = df.sample(min(30, len(df)))
                    for _, r in s.iterrows(): records.append((str(r.iloc[0]), "unknown"))
            else:
                raw_data = []
                if ds["type"] == "JSON":
                    with open(ds["path"], "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                else:
                    with open(ds["path"], "r", encoding="utf-8") as f:
                        raw_data = [json.loads(l) for l in f if l.strip()]
                
                if not isinstance(raw_data, list): raw_data = [raw_data]
                s = random.sample(raw_data, min(30, len(raw_data)))
                for rec in s:
                    txt = rec.get("text") or rec.get("User Instruction") or rec.get("Attacker Instruction") or str(rec)
                    lbl = "malicious" if rec.get("label") == 1 or rec.get("Modifed") == 1 else "benign"
                    records.append((txt, lbl))
        except Exception as e:
            print(f"Error sampling {name}: {e}")
            continue

        tp = fp = tn = fn = 0
        times = []
        
        try:
            for text, true_label in records:
                t0 = time.time()
                final_pred = "benign"
                l1_score = 0.0
                l2_mse = 0.0
                
                try:
                    # L1
                    try:
                        res1 = json.loads(l1_tool._run(text=text))
                        l1_score = res1["security_metadata"]["threat_score"]
                    except SecurityException:
                        final_pred = "malicious"
                    
                    # PPL
                    try:
                        ppl_tool._run(prompt=text)
                    except SecurityException:
                        final_pred = "malicious"
                        
                    # L2
                    res2 = json.loads(l2_tool._run(agent_action_text=text))
                    l2_mse = res2["score"]
                    
                    # L3
                    res3 = json.loads(l3_tool._run(user_query=text, agent_action=text, l1_score=l1_score, l2_mse=l2_mse))
                    if res3["status"] == "QUARANTINE":
                        final_pred = "malicious"
                except Exception:
                    final_pred = "error"
                    
                ms = round((time.time() - t0) * 1000)
                times.append(ms)
                
                if final_pred == "malicious" and true_label == "malicious": tp += 1
                elif final_pred == "malicious" and true_label == "benign":  fp += 1
                elif final_pred == "benign"    and true_label == "benign":  tn += 1
                elif final_pred == "benign"    and true_label == "malicious": fn += 1

            total = tp + fp + tn + fn
            acc = (tp + tn) / max(total, 1)
            all_results["datasets"][name] = {
                "accuracy": round(acc, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "avg_ms": round(sum(times) / max(len(times), 1))
            }
            
            # Incremental Save
            BENCH_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(BENCH_RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            continue

    _bench_status["status"] = "idle"
    _bench_status["progress"] = 100

@app.route("/api/benchmark/all", methods=["POST"])
def benchmark_all():
    global _bench_status
    if _bench_status["status"] == "running":
        return jsonify({"error": "Benchmark already running"}), 400
    
    _bench_status = {"status": "running", "progress": 0, "current_dataset": "Starting..."}
    threading.Thread(target=_run_full_bench, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/benchmark/status")
def benchmark_status():
    return jsonify(_bench_status)

@app.route("/api/benchmark/results")
def benchmark_results():
    if BENCH_RESULTS_FILE.exists():
        with open(BENCH_RESULTS_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No results yet"}), 404

@app.route("/api/history")
def get_history():
    try:
        conn = sqlite3.connect(HISTORY_DB)
        c = conn.cursor()
        c.execute("SELECT id, timestamp, prompt, agent_id, verdict, avg_risk, total_ms FROM history ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        return jsonify([{
            "id": r[0], "timestamp": r[1], "prompt": r[2], "agent_id": r[3],
            "verdict": r[4], "avg_risk": r[5], "total_ms": r[6]
        } for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/trust/<agent_id>")
def trust_history(agent_id: str):
    try:
        db = BASE_DIR / "final_causal_model.db"
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute(
            "SELECT risk_score, timestamp FROM watchman_trust_history "
            "WHERE agent_id=? ORDER BY timestamp ASC LIMIT 100",
            (agent_id,),
        )
        rows = [{"risk_score": r[0], "timestamp": r[1]} for r in c.fetchall()]
        conn.close()
        return jsonify({"agent_id": agent_id, "history": rows})
    except Exception as e:
        return jsonify({"error": str(e)})


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  KingsGuard Dashboard — http://localhost:8000")
    print("  Auto-preloading models synchronously...")
    print("=" * 60)
    _preload_all()
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 7860)), threaded=True)
