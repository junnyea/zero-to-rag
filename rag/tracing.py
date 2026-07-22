import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

# Track global schema version (as required by R1 / Future considerations)
SCHEMA_VERSION = 1

class TraceEmitter:
    """
    Collects and persists structured trace events for a single question-answering session.
    Implements the REASON -> ACT -> OBSERVE pattern specified in PRD §7 (R1).
    """
    def __init__(self, trace_dir: str = "./traces", trace_keep: int = 200):
        self.trace_dir = trace_dir
        self.trace_keep = trace_keep
        self.trace_id = f"tr_{uuid.uuid4().hex[:8]}"
        self.seq = 0
        self.events: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

        # Ensure trace directory exists
        os.makedirs(self.trace_dir, exist_ok=True)

    def emit(self, phase: str, step: str, detail: str, payload: Optional[Dict[str, Any]] = None, duration_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Emits and records a single trace event.
        phase: REASON | ACT | OBSERVE
        step: Name of the active pipeline step (e.g., "retrieval", "rewrite", "rerank", "generation")
        detail: Brief description of the event
        payload: Arbitrary extra details (without credentials)
        """
        if phase not in ("REASON", "ACT", "OBSERVE"):
            raise ValueError(f"Invalid trace phase: {phase}. Must be REASON, ACT, or OBSERVE.")

        self.seq += 1
        event = {
            "trace_id": self.trace_id,
            "schema_version": SCHEMA_VERSION,
            "seq": self.seq,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "step": step,
            "detail": detail,
            "payload": payload or {},
            "duration_ms": duration_ms
        }
        self.events.append(event)
        return event

    def save_to_disk(self) -> str:
        """
        Appends the collected events for this trace as a single logical transaction or block of lines
        into a daily JSONL file: traces/YYYY-MM-DD.jsonl
        Returns the absolute/relative file path.
        """
        if not self.events:
            return ""

        today_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(self.trace_dir, f"{today_str}.jsonl")

        try:
            # We open with append mode and write each event as a separate line
            with open(file_path, "a", encoding="utf-8") as f:
                for event in self.events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

            # Post-save rotation: clean old traces to respect trace_keep limit (P1 requirement)
            self._rotate_traces()

            return file_path
        except Exception as e:
            # R1: A tracing failure is itself logged but never blocks or alters the answer path.
            print(f"Warning: Failed to save trace {self.trace_id} to disk: {e}")
            return ""

    def _rotate_traces(self) -> None:
        """
        Keeps trace count within limits. Since events are stored in daily files,
        we can count distinct trace_ids in active files and rotate older files
        or entries if needed, but a simpler robust P1 strategy is to rotate daily files
        if total lines/traces across all files exceed trace_keep, or just rotate files older
        than N days if they contain old traces. Let's list all jsonl files, read trace IDs,
        and if we have too many, prune the oldest files.
        """
        try:
            jsonl_files = sorted([
                os.path.join(self.trace_dir, f)
                for f in os.listdir(self.trace_dir)
                if f.endswith(".jsonl")
            ])

            # Gather all distinct trace_ids across all files
            traces_info = [] # List of tuples: (trace_id, file_path, line_count)
            for fpath in jsonl_files:
                # We can do a quick pass to count traces and check boundaries
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        seen_in_file = set()
                        for line in f:
                            try:
                                data = json.loads(line)
                                tid = data.get("trace_id")
                                if tid and tid not in seen_in_file:
                                    seen_in_file.add(tid)
                                    traces_info.append((tid, fpath))
                            except Exception:
                                continue
                except Exception:
                    continue

            # If total distinct trace count exceeds trace_keep, let's remove the oldest daily files
            # to be safe and simple (avoiding complex rewriting of files unless necessary).
            # If a daily file is very old, we can delete it.
            if len(traces_info) > self.trace_keep:
                # To keep it extremely safe and simple, if we have more than trace_keep distinct traces,
                # we can remove the oldest JSONL file if we have multiple files.
                if len(jsonl_files) > 1:
                    os.remove(jsonl_files[0])
        except Exception as e:
            print(f"Warning: Trace rotation failed: {e}")


def load_all_traces(trace_dir: str = "./traces") -> List[Dict[str, Any]]:
    """
    Loads and groups all trace events by trace_id from the trace directory.
    Returns a list of structured trace objects:
    [
        {
            "trace_id": str,
            "timestamp": str (first event timestamp),
            "question": str (extracted from first REASON or query step),
            "strategy": str,
            "reranker": str,
            "events": List[Dict]
        }
    ]
    Sorted newest first.
    """
    if not os.path.exists(trace_dir):
        return []

    try:
        jsonl_files = sorted([
            os.path.join(trace_dir, f)
            for f in os.listdir(trace_dir)
            if f.endswith(".jsonl")
        ], reverse=True) # Newest files first

        traces_map = {}

        for fpath in jsonl_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            tid = event.get("trace_id")
                            if not tid:
                                continue

                            if tid not in traces_map:
                                traces_map[tid] = {
                                    "trace_id": tid,
                                    "timestamp": event.get("timestamp"),
                                    "question": "",
                                    "strategy": "unknown",
                                    "reranker": "unknown",
                                    "events": []
                                }

                            traces_map[tid]["events"].append(event)

                            # Try to extract the original question from the query step
                            if event.get("step") == "query" and event.get("phase") == "REASON":
                                payload = event.get("payload", {})
                                if "question" in payload:
                                    traces_map[tid]["question"] = payload["question"]
                                if "retrieval_strategy" in payload:
                                    traces_map[tid]["strategy"] = payload["retrieval_strategy"]
                                if "reranker" in payload:
                                    traces_map[tid]["reranker"] = payload["reranker"]
                        except Exception:
                            continue
            except Exception:
                continue

        # Convert to list and sort events inside each trace by sequence (seq)
        traces_list = []
        for tid, tdata in traces_map.items():
            tdata["events"].sort(key=lambda x: x.get("seq", 0))
            # Set timestamp to the first event's timestamp
            if tdata["events"]:
                tdata["timestamp"] = tdata["events"][0].get("timestamp")
                # Fallback for question search
                if not tdata["question"]:
                    for ev in tdata["events"]:
                        if "question" in ev.get("payload", {}):
                            tdata["question"] = ev["payload"]["question"]
                            break
                # Fallback for strategy/reranker search
                for ev in tdata["events"]:
                    payload = ev.get("payload", {})
                    if "retrieval_strategy" in payload:
                        tdata["strategy"] = payload["retrieval_strategy"]
                    if "reranker" in payload:
                        tdata["reranker"] = payload["reranker"]

            traces_list.append(tdata)

        # Sort traces by timestamp descending
        traces_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return traces_list
    except Exception as e:
        print(f"Warning: Failed to load traces: {e}")
        return []
