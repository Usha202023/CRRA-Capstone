"""
orchestrator/supervisor.py - CRRA contract renewal review as a LangGraph StateGraph.

    analysis --> policy_check --(hitl_required)--> hitl --> report --> END
                              \\--(otherwise)--------------> report

Run from the project root (contract shim must be running on port 5001):
    python orchestrator/supervisor.py                 # reviews CTR-1004
    python orchestrator/supervisor.py CTR-1004 CTR-1006
"""

import sys
from pathlib import Path

# Make the project root importable so `guardrails` resolves when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
import os  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from typing import Any, TypedDict  # noqa: E402

import anthropic  # noqa: E402
import chromadb  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from guardrails.audit_logger import AuditLogger  # noqa: E402

try:  # optional: load ANTHROPIC_API_KEY from a .env file in the project root
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL = "claude-opus-5"
OUTPUT_CONFIG = {"effort": "medium"}       # temperature is not supported on this model
MAX_TOKENS = 4096
MAX_ROUNDS = 5

CONTRACT_API = "http://localhost:5001/api/contracts/{contract_id}"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "crra_policy"

RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "RIGHTSIZE", "CONSOLIDATE", "TERMINATE"]
CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]

audit = AuditLogger()


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class ContractState(TypedDict, total=False):
    contract_id: str
    contract: dict
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: float
    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str
    final_status: str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fetch_contract(contract_id: str) -> dict:
    url = CONTRACT_API.format(contract_id=contract_id)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Contract API returned {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach the contract API at {url} - is contract_shim.py running?") from e


_collection = None


def get_collection():
    """Open the ChromaDB collection once and reuse it."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def search_policy(query: str, n_results: int = 3) -> list[dict]:
    n_results = max(1, min(int(n_results or 3), 5))
    res = get_collection().query(query_texts=[query], n_results=n_results,
                                 include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "source": meta.get("source"),
            "heading": meta.get("heading"),
            "confidence": round(1 - dist, 3),
            "text": doc,
        })
    return hits


def extract_text(response) -> str:
    """Return the text of the first content block that has a .text attribute."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def approval_band(contract: dict) -> str:
    if contract.get("approval_band"):
        return str(contract["approval_band"]).upper()
    v = int(contract.get("annual_value_inr", 0))
    return "A" if v < 1_000_000 else ("B" if v <= 5_000_000 else "C")


# --------------------------------------------------------------------------
# Tool definitions
# --------------------------------------------------------------------------
TOOLS = [
    {
        "name": "search_policy",
        "description": (
            "Search the Zensar procurement policy knowledge base. Returns the best-matching "
            "policy sections with source file, section heading and text. Use it to find the "
            "rules on approvals, renewals, notice periods, utilisation and uplifts before deciding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language policy question"},
                "n_results": {"type": "integer", "minimum": 1, "maximum": 5,
                              "description": "Number of sections to return (default 3)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Submit the final renewal recommendation. Call exactly once, after checking policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {"type": "string", "enum": RECOMMENDATIONS},
                "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                "rationale": {"type": "string",
                              "description": "2-4 sentences grounded in the contract data and policy"},
                "policy_citation": {"type": "string",
                                    "description": "Source file and section heading, e.g. 'renewals.md > Notice periods'"},
                "estimated_annual_impact_inr": {
                    "type": "number",
                    "description": "Estimated annual saving (positive) or extra cost (negative) in INR",
                },
            },
            "required": ["recommendation", "confidence", "rationale", "policy_citation",
                         "estimated_annual_impact_inr"],
        },
    },
]

SYSTEM_PROMPT = f"""You are the Contract Renewal Analysis Agent for Zensar BizOps.
Review one software/services contract and recommend one of: {", ".join(RECOMMENDATIONS)}.

Process:
1. Use search_policy to look up the procurement rules that apply (approval bands, notice periods,
   seat utilisation, price uplifts, ownership). Search more than once if needed.
2. Base every claim on the contract data provided and the policy text returned. Do not invent rules.
3. Finish by calling submit_recommendation exactly once. Cite the policy file and section you relied on.
Use confidence LOW if the policy is unclear or the data is incomplete.
You have at most {MAX_ROUNDS} rounds."""


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def analysis_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    audit.log("analysis", "start", cid)

    contract = fetch_contract(cid)
    audit.log("analysis", "contract_fetched", cid, vendor=contract.get("vendor"),
              annual_value_inr=contract.get("annual_value_inr"),
              notice_state=contract.get("notice_state"))

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": "Review this contract and submit a renewal recommendation.\n\n"
                   + json.dumps(contract, indent=2),
    }]
    submission: dict | None = None

    for round_no in range(1, MAX_ROUNDS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config=OUTPUT_CONFIG,
        )
        text = extract_text(response)
        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        audit.log("analysis", "model_round", cid, round=round_no, stop_reason=response.stop_reason,
                  tools_called=[t.name for t in tool_uses], text=text[:500])

        # Keep the assistant turn exactly as returned (including any thinking blocks)
        messages.append({"role": "assistant", "content": response.content})

        if not tool_uses:
            messages.append({"role": "user",
                             "content": "Please call submit_recommendation with your final answer."})
            continue

        tool_results = []
        for tu in tool_uses:
            if tu.name == "submit_recommendation":
                submission = dict(tu.input)
                audit.log("analysis", "recommendation_submitted", cid, round=round_no, **submission)
                break
            if tu.name == "search_policy":
                try:
                    hits = search_policy(**tu.input)
                    audit.log("analysis", "search_policy", cid, query=tu.input.get("query"),
                              results=[f"{h['source']} > {h['heading']} ({h['confidence']})" for h in hits])
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                         "content": json.dumps(hits, ensure_ascii=False)})
                except Exception as e:  # report tool errors back to the model
                    audit.log("analysis", "tool_error", cid, tool=tu.name, error=str(e))
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                         "content": f"Error: {e}", "is_error": True})
            else:
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                     "content": f"Unknown tool '{tu.name}'", "is_error": True})

        if submission is not None:
            break
        messages.append({"role": "user", "content": tool_results})

    if submission is None:
        audit.log("analysis", "no_recommendation", cid, rounds=MAX_ROUNDS)
        submission = {
            "recommendation": "NO_RECOMMENDATION",
            "confidence": "LOW",
            "rationale": f"The analysis agent did not submit a recommendation within {MAX_ROUNDS} rounds.",
            "policy_citation": "",
            "estimated_annual_impact_inr": 0,
        }

    return {
        "contract": contract,
        "recommendation": str(submission.get("recommendation", "")).upper(),
        "confidence": str(submission.get("confidence", "LOW")).upper(),
        "rationale": submission.get("rationale", ""),
        "policy_citation": submission.get("policy_citation", ""),
        "estimated_annual_impact_inr": float(submission.get("estimated_annual_impact_inr") or 0),
    }


def policy_check_node(state: ContractState) -> dict:
    """Deterministic guardrail - no model call. Collects every reason that needs a human."""
    cid = state["contract_id"]
    contract = state.get("contract", {})
    reasons: list[str] = []

    band = approval_band(contract)
    if band in ("B", "C"):
        reasons.append(f"Approval band {band} (annual value INR {contract.get('annual_value_inr', 0):,})")
    if str(contract.get("notice_state", "")).upper() == "INSIDE_WINDOW":
        reasons.append(f"Inside notice window (notice deadline {contract.get('notice_deadline')})")
    if state.get("recommendation") == "TERMINATE":
        reasons.append("Recommendation is TERMINATE")
    if state.get("confidence") == "LOW":
        reasons.append("Model confidence is LOW")
    if str(contract.get("owner", "")).strip().upper() == "UNASSIGNED":
        reasons.append("Contract owner is UNASSIGNED")

    result = {"hitl_required": bool(reasons), "hitl_reason": "; ".join(reasons)}
    audit.log("policy_check", "evaluated", cid, approval_band=band, **result)
    return result


def hitl_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    c = state.get("contract", {})
    audit.log("hitl", "review_requested", cid, hitl_reason=state.get("hitl_reason"))

    print("\n" + "=" * 70)
    print(f"  HUMAN REVIEW REQUIRED - {cid} ({c.get('vendor')})")
    print("=" * 70)
    print(f"  Recommendation : {state.get('recommendation')}  (confidence {state.get('confidence')})")
    print(f"  Annual impact  : INR {state.get('estimated_annual_impact_inr', 0):,.0f}")
    print(f"  Rationale      : {state.get('rationale')}")
    print(f"  Policy         : {state.get('policy_citation')}")
    print("  Review reasons :")
    for r in (state.get("hitl_reason") or "").split("; "):
        print(f"    - {r}")
    print("=" * 70)

    try:
        while True:
            answer = input("Approve this recommendation? [y/n]: ").strip().lower()
            if answer in ("y", "yes", "n", "no"):
                break
            print("Please type y or n.")
        approved = answer.startswith("y")
        approver = ""
        if approved:
            while not approver:
                approver = input("Approver name: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received - treating as rejected.")
        approved, approver = False, ""

    audit.log("hitl", "decision", cid, approved=approved, approver=approver or None)
    return {"hitl_approved": approved, "approver": approver}


def report_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    rec = state.get("recommendation") or "NO_RECOMMENDATION"

    if not state.get("hitl_required"):
        final_status = f"{rec}_AUTO"
    elif state.get("hitl_approved"):
        final_status = f"{rec}_APPROVED"
    else:
        final_status = "ON_HOLD_REJECTED"

    audit.log("report", "final", cid, final_status=final_status, recommendation=rec,
              confidence=state.get("confidence"),
              estimated_annual_impact_inr=state.get("estimated_annual_impact_inr"),
              policy_citation=state.get("policy_citation"),
              hitl_required=state.get("hitl_required"), hitl_reason=state.get("hitl_reason"),
              approver=state.get("approver") or None)

    print(f"\n>>> {cid}: {final_status}")
    return {"final_status": final_status}


def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ContractState)
    graph.add_node("analysis", analysis_node)
    graph.add_node("policy_check", policy_check_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "policy_check")
    graph.add_conditional_edges("policy_check", route_after_policy_check,
                                {"hitl": "hitl", "report": "report"})
    graph.add_edge("hitl", "report")
    graph.add_edge("report", END)
    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (set it in your shell or a .env file in the project root).")

    contract_ids = sys.argv[1:] or ["CTR-1004"]
    app = build_graph()
    audit.log("supervisor", "run_start", None, contract_ids=contract_ids, model=MODEL)

    summary = []
    for cid in contract_ids:
        try:
            final = app.invoke({"contract_id": cid.upper()})
            summary.append((cid.upper(), final.get("final_status")))
        except Exception as e:
            audit.log("supervisor", "error", cid.upper(), error=str(e))
            summary.append((cid.upper(), f"ERROR: {e}"))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    for cid, status in summary:
        print(f"  {cid:<10} {status}")
    print(f"  Audit trail: {audit.log_path}")
    print("=" * 70)
    audit.log("supervisor", "run_end", None, results=dict(summary))


if __name__ == "__main__":
    main()