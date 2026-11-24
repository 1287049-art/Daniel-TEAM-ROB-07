import copy
import datetime
import io
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# CAREER VENTURE LAB - STAGE 1 DASHBOARD
# Pathway: ROB-3 (AP Calculus | Engineering / Robotics - Robotics kit & repair studio)
# ==============================================================================

st.set_page_config(
    page_title="Daniel's Robotics Kit & Repair Studio",
    page_icon="🤖",
    layout="wide",
)


def money(val: Any) -> Decimal:
    """Convert numeric input to exact 2-decimal USD Decimal."""
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt_usd(val: Any) -> str:
    d = money(val)
    sign = "-" if d < 0 else ""
    return f"{sign}${abs(d):,.2f}"


def get_initial_state() -> Dict[str, Any]:
    """Return clean post-startup opening state for ROB-3."""
    return {
        "team_id": "TEAM-ROB-07",
        "course": "AP Calculus",
        "pathway_id": "ROB-3",
        "business_name": "Robotics kit and repair studio",
        "snapshot_id": "SNAP-ROB3-001",
        "state_version": 1,
        "simulation_date": "2027-01-01",
        "base_period": "monthly",
        "working_days": 22,
        "currency": "USD",
        "level_by_skill": {
            "constrained_optimization": "Advanced",
            "rate_integration": "Advanced",
        },
        "assumptions": [
            "All values are fictional teaching defaults from ROB-3.",
            "Linear demand curve q(p) = 2*D0 - (D0/p0)*p on 0 <= p <= 2*p0.",
            "Components are expensed when attempted/consumed; cash leaves on restock.",
            "No automatic backlog; unmet demand is recorded as lost demand.",
            "Simplified accrual with immediate payment; taxes/depreciation disabled at baseline.",
        ],
        "sources": [
            "Career Venture Lab Student Handbook (November 2026) - ROB Package & AP Calculus Adapter"
        ],
        "offerings": [
            {
                "offering_id": "OFF-KIT-01",
                "name": "Introductory robot kit",
                "unit": "kit",
                "price": "120.00",
                "demand_rule": "q(p) = max(0, 80 - (1/3)*p)",
                "capacity_rule": "floor(staff_hours / hours_per_unit)",
                "primary": True,
            },
            {
                "offering_id": "OFF-SENS-02",
                "name": "Sensor upgrade (Planned Stage 2)",
                "unit": "upgrade",
                "price": "45.00",
                "demand_rule": "Separate allocation required",
                "capacity_rule": "Separate allocation required",
                "primary": False,
            },
            {
                "offering_id": "OFF-REP-03",
                "name": "Repair service (Planned Stage 2)",
                "unit": "repair",
                "price": "60.00",
                "demand_rule": "Separate allocation required",
                "capacity_rule": "Separate allocation required",
                "primary": False,
            },
        ],
        "funding": {
            "owner_contribution": "12000.00",
            "equipment_assets": "4000.00",  # Tools $2,500 + Test equipment $1,500
            "pre_opening_expense": "0.00",
            "opening_inventory_value": "3000.00",  # 100 sets @ $30
            "opening_cash": "5000.00",
            "retained_earnings": "0.00",
            "loan_debt": "0.00",
        },
        "params": {
            "base_price": "120.00",
            "D0": 40,
            "staff_hours": 90.0,
            "hours_per_unit": 3.0,
            "opening_stock_units": 100,
            "component_unit_cost": "30.00",
            "restock_units": 0,
            "defect_rate": 0.0,
            "refund_amount": "0.00",
            "loan_principal": "0.00",
            "new_loan_cash": "0.00",
            "interest": "0.00",
            "additional_capex": "0.00",
            "new_owner_cash": "0.00",
        },
        "cost_lines": {
            "variable_per_unit": {
                "components_COGS": "30.00",
                "assembly_labor": "20.00",
                "packaging": "10.00",
                "warranty_consumables": "5.00",
            },
            "fixed_monthly": {
                "workspace": "250.00",
                "software": "100.00",
                "equipment_maintenance": "150.00",
                "admin": "100.00",
            },
            "advertising": "0.00",
        },
        "balances": {
            "cash": "5000.00",
            "stock_units": 100,
            "inventory_value": "3000.00",
            "equipment_assets": "4000.00",
            "debt": "0.00",
            "equity": "12000.00",
        },
        "capacity": {
            "derived_monthly_capacity": 30,
            "bottleneck": "staff_hours (90 hrs / 3.0 hrs/kit = 30 kits)",
        },
        "ledger": [],
        "processed_run_ids": [],
        "campaigns": [],
        "decisions": [],
        "random_seed": 2027,
    }


def last_day_of_month(ym: str) -> str:
    year, month = map(int, ym.split("-"))
    if month == 12:
        nxt = datetime.date(year + 1, 1, 1)
    else:
        nxt = datetime.date(year, month + 1, 1)
    return (nxt - datetime.timedelta(days=1)).isoformat()


def run_monthly_step(
    state: Dict[str, Any],
    sim_month: str,
    requested: int,
    run_id: str,
    capacity_override: int | None = None,
    ad_cost_override: Decimal | None = None,
    restock_override: int | None = None,
    defect_override: float | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Execute one deterministic ROB-3 monthly simulation step.
    Enforces idempotent run_id protection and exact ROB inventory/cash rules.
    """
    if run_id in state["processed_run_ids"]:
        return False, f"Duplicate run_id '{run_id}' rejected: no duplicate transactions created.", {}

    if requested < 0:
        return False, "Requested demand must be >= 0.", {}

    p = state["params"]
    price = money(p["base_price"])
    staff_hours = float(p["staff_hours"])
    hours_per_unit = float(p["hours_per_unit"])
    if hours_per_unit <= 0:
        return False, "hours_per_unit must be > 0.", {}

    defect_rate = float(defect_override if defect_override is not None else p["defect_rate"])
    if not (0.0 <= defect_rate <= 1.0):
        return False, "defect_rate must be in [0, 1].", {}

    restock_units = int(restock_override if restock_override is not None else p["restock_units"])
    if restock_units < 0:
        return False, "restock_units must be >= 0.", {}

    comp_unit_cost = money(p["component_unit_cost"])
    v_lines = state["cost_lines"]["variable_per_unit"]
    non_comp_var_unit = (
        money(v_lines["assembly_labor"])
        + money(v_lines["packaging"])
        + money(v_lines["warranty_consumables"])
    )
    variable_per_unit = comp_unit_cost + non_comp_var_unit
    fixed_expense = sum(money(v) for v in state["cost_lines"]["fixed_monthly"].values())
    marketing_expense = (
        money(ad_cost_override)
        if ad_cost_override is not None
        else money(state["cost_lines"]["advertising"])
    )

    refunds = money(p["refund_amount"])
    interest = money(p["interest"])
    principal = money(p["loan_principal"])
    additional_capex = money(p["additional_capex"])
    new_owner_cash = money(p["new_owner_cash"])
    loan_cash = money(p["new_loan_cash"])

    opening_cash = money(state["balances"]["cash"])
    opening_stock = int(state["balances"]["stock_units"])

    # Capacity rule
    derived_cap = math.floor(staff_hours / hours_per_unit)
    effective_cap = int(capacity_override) if capacity_override is not None else derived_cap
    available_stock = opening_stock + restock_units

    attempted = min(int(requested), effective_cap, available_stock)
    failed = math.floor(attempted * defect_rate)
    sold = attempted - failed
    lost_demand = int(requested) - attempted

    # Accounting calculations (ROB-specific rules)
    gross_revenue = price * Decimal(sold)
    net_revenue = gross_revenue - refunds

    # Variable expense recognized on ALL attempted units (including failed builds)
    non_component_var_cost = non_comp_var_unit * Decimal(attempted)
    variable_expense = variable_per_unit * Decimal(attempted)

    total_operating_expense = variable_expense + fixed_expense + marketing_expense
    operating_profit = net_revenue - total_operating_expense
    net_profit = operating_profit - interest

    # Cash flow: components only drain cash on restock, NOT at consumption
    restock_cash_outflow = comp_unit_cost * Decimal(restock_units)
    ending_cash = (
        opening_cash
        + net_revenue
        - non_component_var_cost
        - restock_cash_outflow
        - fixed_expense
        - marketing_expense
        - interest
        - principal
        - additional_capex
        + new_owner_cash
        + loan_cash
    )

    ending_stock = available_stock - attempted
    ending_inventory_value = comp_unit_cost * Decimal(ending_stock)

    tx_date = last_day_of_month(sim_month)
    new_version = state["state_version"] + 1

    # Record dated ledger entries
    entries = []
    if sold > 0:
        entries.append(
            {
                "transaction_id": f"{run_id}-REV",
                "date": tx_date,
                "event_type": "sale",
                "category": "sales_revenue",
                "amount": str(net_revenue),
                "quantity": sold,
                "unit_price": str(price),
                "offering_id": "OFF-KIT-01",
                "campaign_id": "",
                "financing_type": "",
                "state_version": new_version,
            }
        )
    if attempted > 0:
        entries.append(
            {
                "transaction_id": f"{run_id}-VAR",
                "date": tx_date,
                "event_type": "expense",
                "category": "variable_production_expense",
                "amount": str(-variable_expense),
                "quantity": attempted,
                "unit_price": str(variable_per_unit),
                "offering_id": "OFF-KIT-01",
                "campaign_id": "",
                "financing_type": "",
                "state_version": new_version,
            }
        )
    if restock_units > 0:
        entries.append(
            {
                "transaction_id": f"{run_id}-RST",
                "date": tx_date,
                "event_type": "inventory_purchase",
                "category": "component_restock_cash",
                "amount": str(-restock_cash_outflow),
                "quantity": restock_units,
                "unit_price": str(comp_unit_cost),
                "offering_id": "OFF-KIT-01",
                "campaign_id": "",
                "financing_type": "",
                "state_version": new_version,
            }
        )
    entries.append(
        {
            "transaction_id": f"{run_id}-FIX",
            "date": tx_date,
            "event_type": "expense",
            "category": "fixed_monthly_expense",
            "amount": str(-fixed_expense),
            "quantity": 1,
            "unit_price": str(fixed_expense),
            "offering_id": "",
            "campaign_id": "",
            "financing_type": "",
            "state_version": new_version,
        }
    )
    if marketing_expense > 0:
        entries.append(
            {
                "transaction_id": f"{run_id}-MKT",
                "date": tx_date,
                "event_type": "marketing",
                "category": "own_campaign_expense",
                "amount": str(-marketing_expense),
                "quantity": 1,
                "unit_price": str(marketing_expense),
                "offering_id": "OFF-KIT-01",
                "campaign_id": f"CAMP-{sim_month}",
                "financing_type": "",
                "state_version": new_version,
            }
        )
    if loan_cash > 0:
        entries.append(
            {
                "transaction_id": f"{run_id}-LOAN",
                "date": tx_date,
                "event_type": "financing",
                "category": "loan_proceeds",
                "amount": str(loan_cash),
                "quantity": 1,
                "unit_price": str(loan_cash),
                "offering_id": "",
                "campaign_id": "",
                "financing_type": "loan",
                "state_version": new_version,
            }
        )

    state["ledger"].extend(entries)
    state["processed_run_ids"].append(run_id)
    state["state_version"] = new_version
    state["simulation_date"] = tx_date
    state["snapshot_id"] = f"SNAP-ROB3-{new_version:03d}"

    state["balances"]["cash"] = str(ending_cash)
    state["balances"]["stock_units"] = ending_stock
    state["balances"]["inventory_value"] = str(ending_inventory_value)
    state["balances"]["equipment_assets"] = str(
        money(state["balances"]["equipment_assets"]) + additional_capex
    )
    state["balances"]["debt"] = str(
        money(state["balances"]["debt"]) + loan_cash - principal
    )
    state["balances"]["equity"] = str(
        money(state["balances"]["equity"]) + new_owner_cash + net_profit
    )

    summary = {
        "month": sim_month,
        "requested": requested,
        "capacity": effective_cap,
        "attempted": attempted,
        "failed": failed,
        "sold": sold,
        "lost_demand": lost_demand,
        "revenue": net_revenue,
        "variable_expense": variable_expense,
        "fixed_expense": fixed_expense,
        "marketing_expense": marketing_expense,
        "total_expense": total_operating_expense,
        "profit": net_profit,
        "closing_cash": ending_cash,
        "ending_stock": ending_stock,
        "ending_inventory_value": ending_inventory_value,
    }
    return True, f"Recorded {sim_month} (state_version -> v{new_version}).", summary


# ==============================================================================
# AP CALCULUS ENGINE (PATHWAY ROB-3)
# ==============================================================================


def solve_ap_calculus(
    D0: float, p0: float, v: float, F: float, cap: int
) -> Dict[str, Any]:
    """
    Deterministic AP Calculus optimization solver:
    Demand q(p) = a - b*p on 0 <= p <= 2*p0, where a = 2*D0, b = D0/p0.
    """
    a = 2.0 * D0
    b = D0 / p0 if p0 > 0 else 0.0
    p_max = 2.0 * p0

    # Unconstrained profit P_unc(p) = (p - v)*(a - b*p) - F
    # P_unc'(p) = a + b*v - 2*b*p = 0 => p* = (a + b*v) / (2*b)
    p_unc_star = (a + b * v) / (2.0 * b) if b > 0 else p0
    p_unc_star = min(max(0.0, p_unc_star), p_max)
    q_unc_star = max(0.0, a - b * p_unc_star)
    prof_unc_star = (p_unc_star - v) * q_unc_star - F

    # Capacity corner: a - b*p = cap => p_corner = (a - cap)/b
    p_corner = (a - cap) / b if b > 0 else 0.0
    p_corner_in_domain = 0.0 <= p_corner <= p_max

    def eval_constrained(price_val: float) -> Tuple[float, int, float, float]:
        q_raw = max(0.0, a - b * price_val)
        s_cont = min(float(cap), q_raw)
        s_int = min(int(cap), math.floor(q_raw))
        prof_cont = (price_val - v) * s_cont - F
        prof_int = (price_val - v) * s_int - F
        return s_cont, s_int, prof_cont, prof_int

    candidates = [0.0, p_max]
    if p_corner_in_domain:
        candidates.append(p_corner)
    if (not p_corner_in_domain) or (p_unc_star >= p_corner):
        candidates.append(p_unc_star)

    best_p = candidates[0]
    best_prof_cont = -1e18
    for c in candidates:
        _, _, pc, _ = eval_constrained(c)
        if pc > best_prof_cont:
            best_prof_cont = pc
            best_p = c

    s_cont_opt, s_int_opt, prof_cont_opt, prof_int_opt = eval_constrained(best_p)

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "p_unc_star": p_unc_star,
        "q_unc_star": q_unc_star,
        "prof_unc_star": prof_unc_star,
        "p_corner": p_corner,
        "p_corner_in_domain": p_corner_in_domain,
        "p_con_star": best_p,
        "s_cont_opt": s_cont_opt,
        "s_int_opt": s_int_opt,
        "prof_cont_opt": prof_cont_opt,
        "prof_int_opt": prof_int_opt,
        "unconstrained_is_infeasible": p_corner_in_domain and (p_unc_star < p_corner),
    }


def solve_rate_integral(
    r0: float, r1: float, c0: float, c1: float, t_end: float
) -> Dict[str, float]:
    """
    Exact definite integral of linear daily rates r(t) = r0 + r1*t and c(t) = c0 + c1*t
    over [0, t_end].
    """
    rev = r0 * t_end + 0.5 * r1 * (t_end**2)
    cost = c0 * t_end + 0.5 * c1 * (t_end**2)
    profit = rev - cost
    return {"rev": rev, "cost": cost, "profit": profit}


# ==============================================================================
# AUTOMATED VERIFICATION SUITE
# ==============================================================================


def run_all_stage1_tests() -> List[Dict[str, str]]:
    results = []

    # Test 1: Monthly Fixture Replay
    s = get_initial_state()
    fixture = [
        ("2027-01", 40, 30, Decimal("0")),
        ("2027-02", 44, 30, Decimal("100")),
        ("2027-03", 36, 22, Decimal("0")),
    ]
    expected = [
        (30, Decimal("3600.00"), Decimal("2550.00"), Decimal("1050.00"), Decimal("6950.00"), 70),
        (30, Decimal("3600.00"), Decimal("2650.00"), Decimal("950.00"), Decimal("8800.00"), 40),
        (22, Decimal("2640.00"), Decimal("2030.00"), Decimal("610.00"), Decimal("10070.00"), 18),
    ]
    fixture_ok = True
    detail_msgs = []
    for idx, (m, req, cap_ov, ad) in enumerate(fixture):
        ok, _, summ = run_monthly_step(
            s, m, req, f"FIX-{m}", capacity_override=cap_ov, ad_cost_override=ad
        )
        exp_sold, exp_rev, exp_exp, exp_prof, exp_cash, exp_stock = expected[idx]
        if not (
            ok
            and summ["sold"] == exp_sold
            and summ["revenue"] == exp_rev
            and summ["total_expense"] == exp_exp
            and summ["profit"] == exp_prof
            and summ["closing_cash"] == exp_cash
            and summ["ending_stock"] == exp_stock
        ):
            fixture_ok = False
        detail_msgs.append(
            f"{m}: Sold={summ['sold']}/{exp_sold}, Profit={fmt_usd(summ['profit'])}/{fmt_usd(exp_prof)}, Cash={fmt_usd(summ['closing_cash'])}/{fmt_usd(exp_cash)}, Stock={summ['ending_stock']}/{exp_stock}"
        )
    results.append(
        {
            "Test": "1. ROB Monthly Fixture Replay (3 months)",
            "Status": "PASS" if fixture_ok else "FAIL",
            "Details": " | ".join(detail_msgs),
        }
    )

    # Test 2: ROB Business-Specific Inventory vs Cash Test
    s2 = get_initial_state()
    s2["balances"]["stock_units"] = 10
    s2["balances"]["inventory_value"] = "300.00"
    ok2, _, summ2 = run_monthly_step(s2, "2027-01", 40, "ROB-SPEC-1", defect_override=0.0)
    t2_ok = (
        ok2
        and summ2["sold"] == 10
        and summ2["revenue"] == Decimal("1200.00")
        and summ2["total_expense"] == Decimal("1250.00")
        and summ2["profit"] == Decimal("-50.00")
        and summ2["ending_stock"] == 0
        and summ2["closing_cash"] == Decimal("5250.00")
    )
    results.append(
        {
            "Test": "2. ROB Stock-Limit & Cash vs. Expense Check",
            "Status": "PASS" if t2_ok else "FAIL",
            "Details": f"Sold={summ2['sold']}, Rev={fmt_usd(summ2['revenue'])}, Exp={fmt_usd(summ2['total_expense'])}, Profit={fmt_usd(summ2['profit'])}, ClosingCash={fmt_usd(summ2['closing_cash'])}, Stock={summ2['ending_stock']}",
        }
    )

    # Test 3: AP Calculus Course Acceptance Check
    calc = solve_ap_calculus(D0=60, p0=30, v=10, F=500, cap=40)
    integ = solve_rate_integral(100, 20, 60, 5, 10)
    t3_ok = (
        abs(calc["p_con_star"] - 40.0) < 1e-6
        and abs(calc["prof_cont_opt"] - 700.0) < 1e-6
        and abs(calc["p_unc_star"] - 35.0) < 1e-6
        and calc["unconstrained_is_infeasible"]
        and abs(integ["profit"] - 1150.0) < 1e-6
    )
    results.append(
        {
            "Test": "3. AP Calculus Constrained Optimum & Integral Fixture",
            "Status": "PASS" if t3_ok else "FAIL",
            "Details": f"Constrained p*=${calc['p_con_star']:.2f} (Profit=${calc['prof_cont_opt']:.2f}) vs Infeasible Unconstrained p*=${calc['p_unc_star']:.2f}; Integral Net Profit=${integ['profit']:.2f}",
        }
    )

    # Test 4: Zero Sales & Duplicate Run Protection
    s4 = get_initial_state()
    ok4a, _, summ4 = run_monthly_step(s4, "2027-01", 0, "ZERO-RUN")
    ok4b, msg4b, _ = run_monthly_step(s4, "2027-01", 0, "ZERO-RUN")
    t4_ok = (
        ok4a
        and summ4["revenue"] == Decimal("0.00")
        and summ4["variable_expense"] == Decimal("0.00")
        and summ4["profit"] == Decimal("-600.00")
        and (not ok4b)
    )
    results.append(
        {
            "Test": "4. Zero-Sales Edge Case & Duplicate run_id Guard",
            "Status": "PASS" if t4_ok else "FAIL",
            "Details": f"Zero-sales Profit={fmt_usd(summ4['profit'])}; Duplicate guard: {msg4b}",
        }
    )

    return results


# ==============================================================================
# INITIALIZE SESSION STATE
# ==============================================================================

if "sim_state" not in st.session_state:
    st.session_state.sim_state = get_initial_state()

state = st.session_state.sim_state

# ==============================================================================
# HEADER & PERSISTENT AP CALCULUS MATH PANEL
# ==============================================================================

st.title("Daniel's Robotics Kit & Repair Studio")
st.caption(
    f"Career Venture Lab • ROB-3 AP Calculus • Team: **{state['team_id']}** • "
    f"Last saved version: **v{state['state_version']}** • Simulation date: **{state['simulation_date']}**"
)
st.markdown(
    "Use the tabs below to run your robotics business. Start with **Overview**, change settings in **Setup**, "
    "study the math in **Calculus**, record each month in **Monthly Run**, and back up your work in **Save & Load**."
)

# Top KPI strip
b = state["balances"]
p = state["params"]
derived_cap = math.floor(float(p["staff_hours"]) / float(p["hours_per_unit"]))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cash available", fmt_usd(b["cash"]))
k2.metric("Robot kits in stock", f"{b['stock_units']} sets")
k3.metric("Build limit each month", f"{derived_cap} kits")
k4.metric("Current kit price", fmt_usd(p["base_price"]))
k5.metric("Saved version", f"v{state['state_version']}")

# Persistent Math Skills & Challenge Level Panel (Visible from Overview per handbook)
with st.expander("Optional: AP Calculus challenge settings", expanded=False):
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        challenge_level = st.selectbox(
            "Challenge Level (Teacher-Approved)",
            ["Foundation", "Applied", "Advanced"],
            index=2,
            help="Changing level changes active equations, tasks, graphs, and interpretation questions.",
        )
        state["level_by_skill"]["constrained_optimization"] = challenge_level
        st.markdown(
            f"""
            - **Course:** AP Calculus (`ROB-3`)
            - **Active Level:** `{challenge_level}`
            - **What you are practicing:** finding the best price and understanding accumulated rates
            - **Price range:** $0$ to ${2 * float(p['base_price']):.2f}
            - **Monthly build limit:** {derived_cap} kits
            """
        )
    with m_col2:
        if challenge_level == "Foundation":
            st.info(
                "**Foundation Mode Task:** Build unconstrained profit $P(p) = (p - v)(a - bp) - F$, "
                "take the derivative $P'(p) = a + bv - 2bp$, and find the critical point $p^* = \frac{a + bv}{2b}$ "
                "and endpoints ($p=0, p=2p_0$)."
            )
            st.markdown(
                "**Interpretation Prompt:** What are the units of $P'(p)$, and what does $P'(p) = 0$ tell you about marginal revenue vs. marginal cost?"
            )
        elif challenge_level == "Applied":
            st.info(
                "**Applied Mode Task:** Enforce the workshop capacity cap $s(p) = \min(\text{cap}, \max(0, a - bp))$. "
                "Analyze the piecewise derivative ($P'(p) = \text{cap}$ on the capped branch; $P'(p) = a + bv - 2bp$ on the demand branch), "
                "check the non-differentiable corner $p = (a - \text{cap})/b$, and compare continuous calculus results to rounded whole-kit sales."
            )
            st.markdown(
                "**Interpretation Prompt:** Why is $P'(p)$ undefined at the capacity corner, and why can an unconstrained critical point be infeasible?"
            )
        else:
            st.info(
                "**Advanced Mode Task:** Perform full constrained optimization with sensitivity checks across cost/capacity disruptions, "
                "AND compute accumulated campaign revenue, cost, and profit via definite integration $\int_0^T [r(t) - c(t)]\,dt$."
            )
            st.markdown(
                "**Interpretation Prompt:** How does a 15% component cost increase shift the optimal price $p^*$, and what do the units of $\int_0^{10} r(t)\,dt$ represent?"
            )

# ==============================================================================
# NAVIGATION TABS
# ==============================================================================

tab_overview, tab_inputs, tab_calc, tab_ops, tab_save = st.tabs(
    [
        "1. Overview",
        "2. Setup",
        "3. Calculus",
        "4. Monthly Run",
        "5. Save & Load",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW & SYSTEM MAP
# ------------------------------------------------------------------------------
with tab_overview:
    st.subheader("Welcome to your robotics business")
    st.markdown(
        """
        Your goal is to run a small business that builds and sells robot kits. You can change the price and staffing,
        use calculus to study the most profitable price, record monthly sales, and track your money and inventory.
        **Nothing is recorded in the business ledger until you approve a monthly run.**
        """
    )
    st.info("**A simple way to use this app:** 1) Review your starting resources. 2) Adjust your business settings. 3) Explore the calculus results. 4) Record a month. 5) Export a backup.")

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("How the business system works", expanded=False):
            st.markdown("#### How information moves through the business")
            st.code(
                """
[Marketing Studio] -> [Customer Demand]
        ^                    |
        |                    v
[Customer Feedback] <- [Building Robot Kits]
                             |
                             v
[Your Approval] <- [Money & Inventory Records]
                """,
                language="text",
            )
            st.markdown(
                "The app automatically checks build limits, failed kits, inventory use, and cash totals. "
                "You control the settings and approve each monthly update."
            )

    with c2:
        st.markdown("#### What is included?")
        st.markdown(
            "- **Setup:** starting money, kit price, demand, staff hours, and supplier disruption\n"
            "- **Calculus:** profit, best price, capacity limits, and rate integration\n"
            "- **Monthly Run:** approve sales and review the transaction ledger\n"
            "- **Save & Load:** download or restore a business snapshot"
        )
        with st.expander("Future features (not available yet)", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Stage": "Stage 1 (Current)", "Modules": "Inputs, Funding, AP Calculus Panel, Ledger, Save/Load JSON, Fixture Tests", "Status": "ACTIVE"},
                        {"Stage": "Stage 2 (Planned)", "Modules": "Report Center, AI Role Imports, Costed Marketing Studio", "Status": "NOT YET IMPLEMENTED"},
                        {"Stage": "Stage 3 (Planned)", "Modules": "Practice Assessment Panel & Cross-Course Artifacts", "Status": "NOT YET IMPLEMENTED"},
                        {"Stage": "Stage 4 (Optional)", "Modules": "Fictional Demonstration Storefront (No Real Payments)", "Status": "NOT YET IMPLEMENTED"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

# ------------------------------------------------------------------------------
# TAB 2: INPUTS & STARTUP FUNDING
# ------------------------------------------------------------------------------
with tab_inputs:
    st.subheader("Set up your business")
    st.caption("Change these settings to see how they affect your business. Changes are not recorded as a monthly result until you approve a run.")

    col_f, col_p = st.columns(2)
    with col_f:
        with st.expander("Where the starting money and equipment came from", expanded=False):
            st.markdown("#### Starting resources (fictional example)")
            st.table(
                pd.DataFrame(
                [
                    {"Item": "Owner Contribution (Equity)", "Amount": "$12,000.00", "Accounting Treatment": "Increases cash & equity; NOT sales revenue"},
                    {"Item": "Tools ($2,500) + Test Equipment ($1,500)", "Amount": "$4,000.00", "Accounting Treatment": "Capital asset; not expensed again"},
                    {"Item": "Initial Component Inventory (100 sets @ $30)", "Amount": "$3,000.00", "Accounting Treatment": "Asset; expensed only as kits are attempted"},
                    {"Item": "Post-Startup Spendable Cash", "Amount": "$5,000.00", "Accounting Treatment": "Opening cash available for operations"},
                ]
                )
            )

        with st.expander("Optional challenge: supplier cost increase", expanded=False):
            disruption_on = st.checkbox(
                "Simulate ROB Disruption: Supplier raises component cost +15% ($30.00 -> $34.50) & delays restock 1 week",
                value=False,
            )
        if disruption_on:
            state["params"]["component_unit_cost"] = "34.50"
            state["cost_lines"]["variable_per_unit"]["components_COGS"] = "34.50"
            st.warning("Disruption active: component_unit_cost is now $34.50/set (total variable cost = $69.50/kit).")
        else:
            state["params"]["component_unit_cost"] = "30.00"
            state["cost_lines"]["variable_per_unit"]["components_COGS"] = "30.00"

    with col_p:
        st.markdown("#### Main business settings")
        st.caption("These settings control your price, expected customer demand, available work time, and build quality.")
        new_team = st.text_input("Team name / ID", value=state["team_id"])
        state["team_id"] = new_team.strip() or "TEAM-ROB-07"

        p_price = st.number_input("Base Price per Kit ($)", min_value=1.0, max_value=500.0, value=float(p["base_price"]), step=5.0)
        p_d0 = st.number_input("Expected customer demand per month (kits)", min_value=0, max_value=500, value=int(p["D0"]), step=1)
        p_hours = st.number_input("Staff work hours available each month", min_value=0.0, max_value=1000.0, value=float(p["staff_hours"]), step=3.0)
        p_hpu = st.number_input("Hours needed to build one kit", min_value=0.5, max_value=40.0, value=float(p["hours_per_unit"]), step=0.5)
        p_defect = st.slider("Build failure rate", min_value=0.0, max_value=0.50, value=float(p["defect_rate"]), step=0.01)

        state["params"]["base_price"] = str(money(p_price))
        state["params"]["D0"] = int(p_d0)
        state["params"]["staff_hours"] = float(p_hours)
        state["params"]["hours_per_unit"] = float(p_hpu)
        state["params"]["defect_rate"] = float(p_defect)

        tot_var = money(state["params"]["component_unit_cost"]) + Decimal("35.00")
        st.info(
            f"**Current Unit Variable Cost:** {fmt_usd(tot_var)}/attempted kit "
            f"(Components {fmt_usd(state['params']['component_unit_cost'])} + Labor $20 + Packaging $10 + Warranty $5) | "
            f"**Monthly Fixed Cost:** $600.00"
        )

# ------------------------------------------------------------------------------
# TAB 3: AP CALCULUS OPTIMIZATION & INTEGRALS
# ------------------------------------------------------------------------------
with tab_calc:
    st.subheader("Use calculus to make better business decisions")
    st.caption("This section estimates how price affects demand and profit, then uses definite integrals to add up revenue and costs over time.")

    D0_val = float(state["params"]["D0"])
    p0_val = float(state["params"]["base_price"])
    v_val = float(money(state["params"]["component_unit_cost"]) + Decimal("35.00"))
    F_val = 600.0
    cap_val = math.floor(float(state["params"]["staff_hours"]) / float(state["params"]["hours_per_unit"]))

    calc_res = solve_ap_calculus(D0_val, p0_val, v_val, F_val, cap_val)

    st.info("**How to read this:** The app compares prices and customer demand while respecting how many kits your team can actually build. The best price is an estimate from the current settings, not a guaranteed real-world result.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best price without a build limit", f"${calc_res['p_unc_star']:.2f}", f"Profit: ${calc_res['prof_unc_star']:.2f}")
    m2.metric("Price where demand meets capacity", f"${calc_res['p_corner']:.2f}", f"Cap = {cap_val} kits")
    m3.metric("Best price with your build limit", f"${calc_res['p_con_star']:.2f}", f"Continuous Profit: ${calc_res['prof_cont_opt']:.2f}")
    m4.metric("Estimated profit (whole kits)", f"${calc_res['prof_int_opt']:.2f}", f"Whole kits sold: {calc_res['s_int_opt']}")

    if calc_res["unconstrained_is_infeasible"]:
        st.warning(
            f"**Boundary Check Alert:** The unconstrained derivative peak $p^* = \${calc_res['p_unc_star']:.2f}$ "
            f"requires {calc_res['q_unc_star']:.2f} kits, which exceeds your capacity of {cap_val} kits! "
            f"Therefore, the true constrained maximum occurs at the capacity boundary $p = \${calc_res['p_corner']:.2f}$."
        )

    test_price = st.slider(
        "Try a kit price and see how profit changes ($)",
        min_value=0.0,
        max_value=float(calc_res["p_max"]),
        value=float(round(calc_res["p_con_star"], 2)),
        step=1.0,
    )

    p_grid = np.linspace(0, calc_res["p_max"], 241)
    q_grid = np.maximum(0.0, calc_res["a"] - calc_res["b"] * p_grid)
    s_cont_grid = np.minimum(float(cap_val), q_grid)
    prof_cont_grid = (p_grid - v_val) * s_cont_grid - F_val
    prof_unc_grid = (p_grid - v_val) * q_grid - F_val

    deriv_grid = np.where(
        p_grid < calc_res["p_corner"],
        float(cap_val),
        calc_res["a"] + calc_res["b"] * v_val - 2.0 * calc_res["b"] * p_grid,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Monthly Profit P(p) [Continuous vs. Unconstrained] — Snapshot {state['snapshot_id']} (PROJECTION)",
            "Marginal Profit Derivative P'(p) [$/dollar of price] with Capacity Corner Discontinuity",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=p_grid,
            y=prof_cont_grid,
            name="Constrained Profit P(p)",
            line=dict(color="#0072B2", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=p_grid,
            y=prof_unc_grid,
            name="Unconstrained Profit (Ignores Cap)",
            line=dict(color="#D55E00", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#444444", annotation_text="Break-Even ($0)", row=1, col=1)
    fig.add_vline(
        x=calc_res["p_corner"],
        line_dash="dashdot",
        line_color="#009E73",
        annotation_text=f"Capacity Corner (${calc_res['p_corner']:.2f})",
        row=1,
        col=1,
    )
    fig.add_vline(x=test_price, line_color="#CC79A7", line_width=2, row=1, col=1)

    left_mask = p_grid < calc_res["p_corner"]
    right_mask = p_grid > calc_res["p_corner"]
    fig.add_trace(
        go.Scatter(
            x=p_grid[left_mask],
            y=deriv_grid[left_mask],
            name="P'(p) Capped Branch (= cap)",
            line=dict(color="#009E73", width=3),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=p_grid[right_mask],
            y=deriv_grid[right_mask],
            name="P'(p) Demand Branch",
            line=dict(color="#56B4E9", width=3),
        ),
        row=2,
        col=1,
    )
    if calc_res["p_corner_in_domain"]:
        right_lim = calc_res["a"] + calc_res["b"] * v_val - 2.0 * calc_res["b"] * calc_res["p_corner"]
        fig.add_trace(
            go.Scatter(
                x=[calc_res["p_corner"], calc_res["p_corner"]],
                y=[float(cap_val), right_lim],
                mode="markers",
                name="Corner Jump (P' Undefined)",
                marker=dict(symbol="circle-open", size=10, color="#D55E00", line=dict(width=2)),
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#444444", row=2, col=1)
    fig.update_xaxes(title_text="Kit Price p (USD)", row=2, col=1)
    fig.update_yaxes(title_text="Profit ($/mo)", row=1, col=1)
    fig.update_yaxes(title_text="P'(p) ($ per $1 price)", row=2, col=1)
    fig.update_layout(height=500, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See the detailed price and sales comparison table", expanded=False):
        sample_prices = sorted(
            list(
                {
                    round(p0_val, 2),
                    round(calc_res["p_corner"], 2),
                    round(calc_res["p_unc_star"], 2),
                    round(test_price, 2),
                    180.0,
                }
            )
        )
        rows = []
        for sp in sample_prices:
            if 0 <= sp <= calc_res["p_max"]:
                q_base = max(0.0, calc_res["a"] - calc_res["b"] * sp)
                s_cont = min(float(cap_val), q_base)
                s_int = min(cap_val, math.floor(q_base))
                s_up = min(cap_val, math.floor(q_base * 1.10))
                s_dn = min(cap_val, math.floor(q_base * 0.90))
                rows.append(
                    {
                        "Kit price": f"${sp:.2f}",
                        "Estimated demand": round(q_base, 2),
                        "Kits we can build": round(s_cont, 2),
                        "Whole kits sold": s_int,
                        "Estimated profit": f"${(sp - v_val)*s_cont - F_val:,.2f}",
                        "Profit if demand is 10% higher": f"${(sp - v_val)*s_up - F_val:,.2f}",
                        "Profit if demand is 10% lower": f"${(sp - v_val)*s_dn - F_val:,.2f}",
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if challenge_level == "Advanced":
        st.divider()
        with st.expander("Advanced: add up revenue and costs over time (definite integrals)", expanded=True):
            st.markdown("#### Daily revenue and cost rates")
            st.caption("A rate tells you how many dollars come in or go out each day. The integral adds those daily amounts over the chosen number of days.")
            st.caption(
                "This is a separate classroom math example; it does not change the monthly business ledger. "
                "$\text{Total profit} = \int_0^T [r(t) - c(t)]\,dt$."
            )
            ic1, ic2, ic3, ic4, ic5 = st.columns(5)
            r0 = ic1.number_input("Starting revenue rate ($/day)", value=100.0, step=10.0)
            r1 = ic2.number_input("Revenue rate increase ($/day²)", value=20.0, step=5.0)
            c0 = ic3.number_input("Starting cost rate ($/day)", value=60.0, step=10.0)
            c1 = ic4.number_input("Cost rate increase ($/day²)", value=5.0, step=1.0)
            t_end = ic5.number_input("Number of days", min_value=1.0, max_value=60.0, value=10.0, step=1.0)

            integ_res = solve_rate_integral(r0, r1, c0, c1, t_end)
            ir1, ir2, ir3 = st.columns(3)
            ir1.metric("Total revenue", f"${integ_res['rev']:,.2f}")
            ir2.metric("Total cost", f"${integ_res['cost']:,.2f}")
            ir3.metric("Total profit", f"${integ_res['profit']:,.2f}")

            t_grid = np.linspace(0, t_end, 100)
            r_t = r0 + r1 * t_grid
            c_t = c0 + c1 * t_grid
            fig_int = go.Figure()
            fig_int.add_trace(go.Scatter(x=t_grid, y=r_t, name="Revenue per day", line=dict(color="#0072B2", width=3)))
            fig_int.add_trace(
                go.Scatter(
                    x=t_grid,
                    y=c_t,
                    name="Cost per day",
                    line=dict(color="#D55E00", width=3, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(0, 158, 115, 0.20)",
                )
            )
            fig_int.update_layout(
                title=f"Total profit over {t_end:.0f} days: ${integ_res['profit']:,.2f}",
                xaxis_title="Day",
                yaxis_title="Dollars per day",
                height=340,
            )
            st.plotly_chart(fig_int, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: OPERATIONS, LEDGER & FIXTURE TESTS
# ------------------------------------------------------------------------------
with tab_ops:
    st.subheader("Record this month's business activity")
    st.caption("Enter what you expect to happen this month, review the result, and approve it to update your cash, inventory, and ledger.")

    op_col1, op_col2 = st.columns(2)
    with op_col1:
        st.markdown("#### 1. Enter this month's activity")
        with st.form("advance_month_form"):
            sim_month = st.text_input("Month (YYYY-MM)", value="2027-01")
            req_units = st.number_input("How many kits customers want", min_value=0, max_value=500, value=int(state["params"]["D0"]), step=1)
            restock_in = st.number_input("Additional component sets purchased", min_value=0, max_value=500, value=0, step=5)
            ad_in = st.number_input("Marketing money spent ($)", min_value=0.0, max_value=10000.0, value=0.0, step=25.0)
            run_id_in = st.text_input("Run reference (prevents duplicate entries)", value=f"RUN-{sim_month}-v{state['state_version']}")
            submitted = st.form_submit_button("Approve month and update business records")

        if submitted:
            ok, msg, summ = run_monthly_step(
                state,
                sim_month=sim_month.strip(),
                requested=int(req_units),
                run_id=run_id_in.strip(),
                ad_cost_override=money(ad_in),
                restock_override=int(restock_in),
            )
            if ok:
                st.success(msg)
                st.json({k: str(v) for k, v in summ.items()})
            else:
                st.error(msg)

    with op_col2:
        with st.expander("2. Technical checks (optional)", expanded=False):
            st.markdown("#### Automated checks and sample-data replay")
            st.caption(
                "Fixture mode uses `sample_data/monthly_fixture.csv` (`capacity_override`) ONLY for verification. "
                "Clicking below runs all ROB-3 and AP Calculus checks."
            )
            if st.button("Run All Stage 1 Verification Tests", type="primary"):
                test_rows = run_all_stage1_tests()
                st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)
    
            if st.button("Load & Replay `monthly_fixture.csv` Into Active Ledger (Resets State First)"):
                st.session_state.sim_state = get_initial_state()
                state = st.session_state.sim_state
                st.warning("FIXTURE-TEST MODE ACTIVE: capacity_override used strictly for sample verification.")
                fixture_rows = [
                    ("2027-01", 40, 30, Decimal("0.00")),
                    ("2027-02", 44, 30, Decimal("100.00")),
                    ("2027-03", 36, 22, Decimal("0.00")),
                ]
                summaries = []
                for m, req, cap_ov, ad in fixture_rows:
                    _, _, sm = run_monthly_step(
                        state, m, req, f"FIXTURE-{m}", capacity_override=cap_ov, ad_cost_override=ad
                    )
                    summaries.append({k: str(v) for k, v in sm.items()})
                st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("#### Your transaction history")
    st.caption("Every approved sale, expense, and inventory purchase is listed here.")
    if state["ledger"]:
        df_ledger = pd.DataFrame(state["ledger"])
        st.dataframe(df_ledger, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Ledger as CSV",
            data=df_ledger.to_csv(index=False).encode("utf-8"),
            file_name=f"{state['team_id']}_ROB3_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )
    else:
        st.info("No transactions recorded yet. Approve a month above or replay the fixture to populate the ledger.")

# ------------------------------------------------------------------------------
# TAB 5: SAVE / LOAD JSON SNAPSHOT (DURABLE PERSISTENCE & VALIDATION)
# ------------------------------------------------------------------------------
with tab_save:
    st.subheader("Save your progress or continue from a backup")
    st.markdown(
        """
        Download a backup of your business so you can continue later. To continue from an earlier backup, upload its JSON file below. The app checks that the file belongs to this ROB-3 project before restoring it.
        """
    )

    snapshot_json_str = json.dumps(state, indent=2)
    st.download_button(
        label=f"Download business backup (version {state['state_version']})",
        data=snapshot_json_str.encode("utf-8"),
        file_name=f"{state['team_id']}_ROB3_snapshot_v{state['state_version']}.json",
        mime="application/json",
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload a previous business backup (.json)", type=["json"])
    if uploaded_file is not None:
        try:
            incoming = json.load(uploaded_file)
            required_keys = {"team_id", "pathway_id", "state_version", "balances", "params", "ledger"}
            missing = required_keys - set(incoming.keys())
            if missing:
                st.error(f"Import rejected: missing required snapshot fields {sorted(list(missing))}.")
            elif incoming.get("team_id") != state["team_id"]:
                st.error(
                    f"Import rejected: Team ID mismatch (expected '{state['team_id']}', got '{incoming.get('team_id')}')."
                )
            elif incoming.get("pathway_id") != "ROB-3":
                st.error(f"Import rejected: Pathway mismatch (expected 'ROB-3', got '{incoming.get('pathway_id')}').")
            elif int(incoming.get("state_version", 0)) < int(state["state_version"]):
                st.error(
                    f"Stale snapshot rejected: incoming version v{incoming.get('state_version')} is older than current v{state['state_version']}."
                )
            else:
                if st.button("Confirm Restore Snapshot"):
                    st.session_state.sim_state = incoming
                    st.success(f"Restored snapshot {incoming.get('snapshot_id')} (v{incoming.get('state_version')}).")
                    st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON file: {exc}")
