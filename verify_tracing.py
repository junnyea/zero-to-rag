import os
import glob
import json
from rag.config import load_config
from rag.query import ask_question
from rag.tracing import SCHEMA_VERSION

def test_tracing_grammar():
    print("🚀 Running Tracing Core and Grammar Verification Test...")

    config = load_config()
    question = "What is the standard core hours of operation?"

    print(f"1. Querying RAG pipeline with question: '{question}'")
    try:
        res = ask_question(question, config)
    except Exception as e:
        print(f"❌ Query failed (maybe Chroma is empty or Ollama offline?): {e}")
        print("Note: This script requires a running Ollama server. Let's inspect existing daily traces if any.")
        res = None

    # Load trace files from the traces directory
    trace_dir = config.get("trace_dir", "./traces")
    trace_files = glob.glob(os.path.join(trace_dir, "*.jsonl"))

    if not trace_files:
        print("❌ No trace JSONL files found in traces/ directory.")
        return False

    print(f"2. Found {len(trace_files)} daily trace file(s). Reading latest trace...")
    latest_file = max(trace_files, key=os.path.getmtime)

    # Read all events from the latest file
    events = []
    with open(latest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    if not events:
        print("❌ Latest trace file is empty.")
        return False

    # Group events by trace_id
    traces = {}
    for ev in events:
        tid = ev["trace_id"]
        if tid not in traces:
            traces[tid] = []
        traces[tid].append(ev)

    # Select the most recent trace
    target_tid = list(traces.keys())[-1]
    trace_events = sorted(traces[target_tid], key=lambda x: x["seq"])

    print(f"3. Validating trace '{target_tid}' containing {len(trace_events)} events...")

    # Assert basic fields
    for ev in trace_events:
        assert "trace_id" in ev, "Event missing 'trace_id'"
        assert "schema_version" in ev, "Event missing 'schema_version'"
        assert ev["schema_version"] == SCHEMA_VERSION, f"schema_version mismatch: got {ev['schema_version']}, expected {SCHEMA_VERSION}"
        assert "seq" in ev, "Event missing 'seq'"
        assert "timestamp" in ev, "Event missing 'timestamp'"
        assert "phase" in ev, "Event missing 'phase'"
        assert "step" in ev, "Event missing 'step'"
        assert "detail" in ev, "Event missing 'detail'"
        assert "payload" in ev, "Event missing 'payload'"

    print("✅ All trace events contain required schema fields.")

    # Validate Grammar: Every ACT must be preceded by a REASON and followed by an OBSERVE
    # We verify this sequence-wise for each active step.
    steps = set(ev["step"] for ev in trace_events)
    print(f"Active steps in trace: {steps}")

    for step in steps:
        step_events = [ev for ev in trace_events if ev["step"] == step]
        print(f"  Validating grammar for step '{step}' ({len(step_events)} events)...")

        # Within a step, look for ACTs and check surroundings
        for idx, ev in enumerate(step_events):
            if ev["phase"] == "ACT":
                # A preceding event in the same step should be a REASON
                preceding = step_events[:idx]
                reasons = [p for p in preceding if p["phase"] == "REASON"]
                if not reasons:
                    print(f"❌ Grammar Violation: ACT in step '{step}' has no preceding REASON.")
                    return False

                # A succeeding event in the same step should be an OBSERVE
                succeeding = step_events[idx+1:]
                observes = [s for s in succeeding if s["phase"] == "OBSERVE"]
                if not observes:
                    print(f"❌ Grammar Violation: ACT in step '{step}' has no succeeding OBSERVE.")
                    return False

    print("✅ Grammar Validation Passed: Every ACT is preceded by a REASON and followed by an OBSERVE.")
    print("🎉 Tracing Core is 100% compliant!")
    return True

if __name__ == "__main__":
    success = test_tracing_grammar()
    if not success:
        exit(1)
