# ─────────────────────────────────────────────────────
# backend/tools/spare_parts.py
#
# TOOL 3 — Spare Parts Lookup Tool
#
# PURPOSE:
#   Queries the spare parts catalog (JSON/CSV) to find
#   parts by machine_id, part_id, issue type, or category.
#   Returns availability, price, supplier, lead time.
#
# USED BY: agent.py, FastAPI /parts endpoint,
#          fault_diagnose.py (part recommendations)
# ─────────────────────────────────────────────────────

import json
import sys
from pathlib import Path
from typing import Optional
from cachetools import TTLCache

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PARTS_JSON_PATH

# ── Cache the catalog in memory (rarely changes) ─────
_parts_cache = TTLCache(maxsize=1, ttl=3600)  # 1 hour TTL


def _load_catalog() -> list:
    """Load spare parts catalog from JSON, with caching."""
    if "catalog" in _parts_cache:
        return _parts_cache["catalog"]

    try:
        with open(PARTS_JSON_PATH, "r") as f:
            data = json.load(f)
        catalog = data.get("spare_parts", [])
        _parts_cache["catalog"] = catalog
        return catalog
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def lookup_parts_by_machine(machine_id: str) -> list:
    """Return all spare parts for a given machine."""
    catalog    = _load_catalog()
    machine_id = machine_id.upper().strip()
    return [p for p in catalog if p["machine_id"] == machine_id]


def lookup_parts_by_ids(part_ids: list) -> list:
    """Return specific parts by their part IDs (used after fault diagnosis)."""
    catalog = _load_catalog()
    id_set  = {pid.upper() for pid in part_ids}
    return [p for p in catalog if p["part_id"].upper() in id_set]


def lookup_parts_by_category(machine_id: str, category: str) -> list:
    """Filter parts by machine and category (e.g. 'Bearings', 'Filters')."""
    parts    = lookup_parts_by_machine(machine_id)
    cat_low  = category.lower()
    return [p for p in parts if cat_low in p["category"].lower()]


def search_parts(query: str, machine_id: Optional[str] = None) -> list:
    """
    Fuzzy text search across part names and categories.

    Args:
        query:      Search term e.g. "bearing", "filter", "seal kit"
        machine_id: Optional machine filter

    Returns:
        Matching parts list
    """
    catalog   = _load_catalog()
    query_low = query.lower()

    results = []
    for part in catalog:
        if machine_id and part["machine_id"] != machine_id.upper():
            continue
        if (query_low in part["name"].lower() or
            query_low in part["category"].lower() or
            query_low in part["part_id"].lower()):
            results.append(part)
    return results


def get_low_stock_parts(machine_id: Optional[str] = None) -> list:
    """Return all parts at or below reorder level — for procurement alerts."""
    catalog = _load_catalog()
    low     = []
    for part in catalog:
        if machine_id and part["machine_id"] != machine_id.upper():
            continue
        if part["stock_qty"] <= part["reorder_level"]:
            low.append({
                **part,
                "stock_status": "OUT_OF_STOCK" if part["stock_qty"] == 0 else "LOW_STOCK",
                "shortage":     max(0, part["reorder_level"] - part["stock_qty"] + 2),
            })
    return low


def format_parts_report(parts: list, title: str = "Spare Parts") -> str:
    """Format a parts list as a human-readable table string."""
    if not parts:
        return "No matching spare parts found."

    lines = [f"=== {title} ===", ""]
    lines.append(f"{'Part ID':<18} {'Name':<40} {'Stock':>6} {'Price (INR)':>12} {'Supplier':<22} {'Lead':>6}")
    lines.append("-" * 110)

    for p in parts:
        stock_str = str(p["stock_qty"]) if p["stock_qty"] > 0 else "OUT"
        lines.append(
            f"{p['part_id']:<18} {p['name'][:38]:<40} {stock_str:>6} "
            f"{p['unit_price']:>12,} {p['supplier'][:20]:<22} {p['lead_time_days']:>4}d"
        )

    total_val = sum(p["unit_price"] * p["stock_qty"] for p in parts)
    lines.append("-" * 110)
    lines.append(f"Total parts shown: {len(parts)}   |   Stock value: ₹{total_val:,}")
    return "\n".join(lines)


def get_parts_for_fault(part_ids: list) -> str:
    """
    Convenience: given a list of part IDs from fault diagnosis,
    return formatted report with availability info.
    """
    if not part_ids:
        return "No specific spare parts identified for this fault."

    parts = lookup_parts_by_ids(part_ids)

    if not parts:
        return f"Parts {part_ids} not found in catalog. Contact procurement."

    lines = ["Recommended Spare Parts:", ""]
    for p in parts:
        avail = f"✓ In stock ({p['stock_qty']} units)" if p["stock_qty"] > 0 else "✗ OUT OF STOCK"
        lines.append(f"  • {p['part_id']} — {p['name']}")
        lines.append(f"    Price: ₹{p['unit_price']:,}  |  {avail}  |  Supplier: {p['supplier']}  |  Lead: {p['lead_time_days']} days")

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    # Test 1 — all parts for CNC
    parts = lookup_parts_by_machine("CNC-M01")
    print(format_parts_report(parts, "CNC-M01 Spare Parts"))

    # Test 2 — fault parts lookup
    print("\n" + get_parts_for_fault(["SP-CNC-T01", "SP-CNC-B01"]))

    # Test 3 — low stock alert
    print("\n=== Low Stock Alert ===")
    low = get_low_stock_parts()
    for p in low:
        print(f"  {p['part_id']:<18} {p['name'][:35]:<35} stock={p['stock_qty']}  status={p['stock_status']}")
