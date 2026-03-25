#!/usr/bin/env python3
"""
Engineering Economics Module for Cloud-based Biometric Sensors

Implements core engineering economics calculations to quantify the wealth
creation potential of cloud-based biometric monitoring systems, including:

  - Time Value of Money (NPV, FV, PV)
  - Internal Rate of Return (IRR)
  - Payback Period (simple and discounted)
  - Return on Investment (ROI)
  - Benefit-Cost Ratio (BCR)
  - Break-Even Analysis
  - Life-Cycle Cost Analysis (LCCA)
  - Depreciation (straight-line and MACRS)
  - Sensitivity Analysis

All monetary values are in USD unless otherwise noted.

References
----------
Engineering Economy, 8th Edition — Blank & Tarquin
Principles of Engineering Economic Analysis, 6th Edition — White et al.
"""

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# 1. Time Value of Money
# ---------------------------------------------------------------------------

def present_value(future_value: float, rate: float, periods: int) -> float:
    """Compute the present value (PV) of a future cash flow.

    Parameters
    ----------
    future_value:
        The cash amount received/paid at the end of *periods*.
    rate:
        Discount rate per period (e.g., 0.10 for 10 %).
    periods:
        Number of compounding periods.

    Returns
    -------
    float
        Present value.

    Examples
    --------
    >>> round(present_value(1000, 0.10, 5), 2)
    620.92
    """
    if rate < 0:
        raise ValueError("Discount rate must be non-negative.")
    if periods < 0:
        raise ValueError("Number of periods must be non-negative.")
    return future_value / (1 + rate) ** periods


def future_value(present_val: float, rate: float, periods: int) -> float:
    """Compute the future value (FV) of a present cash flow.

    Parameters
    ----------
    present_val:
        Cash amount available today.
    rate:
        Interest/growth rate per period.
    periods:
        Number of compounding periods.

    Returns
    -------
    float
        Future value.

    Examples
    --------
    >>> round(future_value(1000, 0.10, 5), 2)
    1610.51
    """
    if rate < 0:
        raise ValueError("Interest rate must be non-negative.")
    if periods < 0:
        raise ValueError("Number of periods must be non-negative.")
    return present_val * (1 + rate) ** periods


def annuity_present_value(payment: float, rate: float, periods: int) -> float:
    """Compute the present value of a uniform series (annuity).

    Uses the P/A factor: PV = A * [(1-(1+i)^-n) / i]

    Parameters
    ----------
    payment:
        Uniform end-of-period payment (annuity amount).
    rate:
        Discount rate per period.
    periods:
        Number of periods.

    Returns
    -------
    float
        Present value of the annuity.

    Examples
    --------
    >>> round(annuity_present_value(500, 0.08, 10), 2)
    3355.04
    """
    if rate <= 0:
        raise ValueError("Discount rate must be positive for annuity calculation.")
    return payment * (1 - (1 + rate) ** (-periods)) / rate


# ---------------------------------------------------------------------------
# 2. Net Present Value (NPV)
# ---------------------------------------------------------------------------

def net_present_value(rate: float, cash_flows: Sequence[float]) -> float:
    """Compute the Net Present Value of a series of cash flows.

    The first element of *cash_flows* is assumed to occur at period 0
    (i.e., today).  Subsequent elements occur at periods 1, 2, …, n.

    Parameters
    ----------
    rate:
        Discount rate per period.
    cash_flows:
        Sequence of cash flows ordered by period.  Outflows are negative.

    Returns
    -------
    float
        NPV of the cash flow series.

    Examples
    --------
    >>> cf = [-50000, 15000, 15000, 15000, 15000, 15000]
    >>> round(net_present_value(0.10, cf), 2)
    6861.8
    """
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


# ---------------------------------------------------------------------------
# 3. Internal Rate of Return (IRR)
# ---------------------------------------------------------------------------

def internal_rate_of_return(
    cash_flows: Sequence[float],
    guess: float = 0.10,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> float:
    """Compute the Internal Rate of Return using Newton-Raphson iteration.

    The IRR is the discount rate that makes the NPV of all cash flows
    equal to zero.

    Parameters
    ----------
    cash_flows:
        Sequence of cash flows (period 0 first).  Outflows are negative.
    guess:
        Initial guess for the IRR (default 10 %).
    tolerance:
        Convergence tolerance.
    max_iterations:
        Maximum number of Newton-Raphson iterations.

    Returns
    -------
    float
        Estimated IRR.

    Raises
    ------
    ValueError
        If the algorithm does not converge within *max_iterations*.

    Examples
    --------
    >>> cf = [-50000, 15000, 15000, 15000, 15000, 15000]
    >>> round(internal_rate_of_return(cf), 4)
    0.1524
    """
    rate = guess
    for _ in range(max_iterations):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        # Derivative of NPV with respect to rate
        d_npv = sum(
            -t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows)
        )
        if d_npv == 0:
            raise ValueError("Derivative is zero; Newton-Raphson cannot continue.")
        new_rate = rate - npv / d_npv
        if abs(new_rate - rate) < tolerance:
            return new_rate
        rate = new_rate
    raise ValueError(
        f"IRR did not converge after {max_iterations} iterations. "
        "Try a different initial guess."
    )


# ---------------------------------------------------------------------------
# 4. Payback Period
# ---------------------------------------------------------------------------

def simple_payback_period(cash_flows: Sequence[float]) -> float:
    """Compute the simple (undiscounted) payback period.

    The payback period is the number of periods required to recover the
    initial investment from cumulative net cash inflows.

    Parameters
    ----------
    cash_flows:
        Sequence of cash flows (period 0 first).  The initial investment
        should be a negative value at index 0.

    Returns
    -------
    float
        Payback period in periods.  Returns ``float('inf')`` if the
        initial investment is never recovered.

    Examples
    --------
    >>> cf = [-50000, 15000, 15000, 15000, 15000, 15000]
    >>> round(simple_payback_period(cf), 2)
    3.33
    """
    cumulative = 0.0
    for t, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            if t == 0:
                return 0.0
            # Interpolate within the period
            previous = cumulative - cf
            fraction = -previous / cf
            return t - 1 + fraction
    return float("inf")


def discounted_payback_period(cash_flows: Sequence[float], rate: float) -> float:
    """Compute the discounted payback period.

    Similar to the simple payback period but uses discounted cash flows.

    Parameters
    ----------
    cash_flows:
        Sequence of cash flows (period 0 first).
    rate:
        Discount rate per period.

    Returns
    -------
    float
        Discounted payback period.  Returns ``float('inf')`` if the
        investment is never recovered on a discounted basis.

    Examples
    --------
    >>> cf = [-50000, 15000, 15000, 15000, 15000, 15000]
    >>> round(discounted_payback_period(cf, 0.10), 2)
    4.26
    """
    cumulative = 0.0
    for t, cf in enumerate(cash_flows):
        discounted_cf = cf / (1 + rate) ** t
        cumulative += discounted_cf
        if cumulative >= 0:
            if t == 0:
                return 0.0
            previous = cumulative - discounted_cf
            fraction = -previous / discounted_cf
            return t - 1 + fraction
    return float("inf")


# ---------------------------------------------------------------------------
# 5. Return on Investment (ROI)
# ---------------------------------------------------------------------------

def return_on_investment(net_benefit: float, total_cost: float) -> float:
    """Compute the Return on Investment (ROI).

    ROI = (Net Benefit / Total Cost) × 100 %

    Parameters
    ----------
    net_benefit:
        Total gains minus total costs (excluding the initial investment).
    total_cost:
        Total initial and operational cost invested.

    Returns
    -------
    float
        ROI expressed as a percentage.

    Raises
    ------
    ValueError
        If *total_cost* is zero.

    Examples
    --------
    >>> round(return_on_investment(25000, 50000), 2)
    50.0
    """
    if total_cost == 0:
        raise ValueError("Total cost cannot be zero.")
    return (net_benefit / total_cost) * 100.0


# ---------------------------------------------------------------------------
# 6. Benefit-Cost Ratio (BCR)
# ---------------------------------------------------------------------------

def benefit_cost_ratio(
    benefits: Sequence[float],
    costs: Sequence[float],
    rate: float,
) -> float:
    """Compute the Benefit-Cost Ratio (BCR) using discounted cash flows.

    BCR = PV(Benefits) / PV(Costs)

    A BCR > 1.0 indicates the project creates net wealth.

    Parameters
    ----------
    benefits:
        Sequence of benefit cash flows ordered by period (period 0 first).
    costs:
        Sequence of cost cash flows (positive values) ordered by period.
    rate:
        Discount rate per period.

    Returns
    -------
    float
        Benefit-Cost Ratio.

    Raises
    ------
    ValueError
        If present value of costs is zero.

    Examples
    --------
    >>> b = [0, 20000, 20000, 20000, 20000, 20000]
    >>> c = [50000, 2000, 2000, 2000, 2000, 2000]
    >>> round(benefit_cost_ratio(b, c, 0.10), 3)
    1.317
    """
    pv_benefits = sum(b / (1 + rate) ** t for t, b in enumerate(benefits))
    pv_costs = sum(c / (1 + rate) ** t for t, c in enumerate(costs))
    if pv_costs == 0:
        raise ValueError("Present value of costs cannot be zero.")
    return pv_benefits / pv_costs


# ---------------------------------------------------------------------------
# 7. Break-Even Analysis
# ---------------------------------------------------------------------------

def break_even_units(
    fixed_costs: float,
    price_per_unit: float,
    variable_cost_per_unit: float,
) -> float:
    """Compute the break-even quantity (units).

    Break-even = Fixed Costs / (Price − Variable Cost per Unit)

    Parameters
    ----------
    fixed_costs:
        Total fixed costs (e.g., hardware, setup, cloud subscription).
    price_per_unit:
        Revenue received per unit sold/deployed.
    variable_cost_per_unit:
        Variable cost incurred per unit.

    Returns
    -------
    float
        Number of units required to break even.

    Raises
    ------
    ValueError
        If contribution margin (price − variable cost) is non-positive.

    Examples
    --------
    >>> round(break_even_units(50000, 300, 50), 2)
    200.0
    """
    contribution_margin = price_per_unit - variable_cost_per_unit
    if contribution_margin <= 0:
        raise ValueError(
            "Price per unit must exceed variable cost per unit (positive contribution margin)."
        )
    return fixed_costs / contribution_margin


# ---------------------------------------------------------------------------
# 8. Life-Cycle Cost Analysis (LCCA)
# ---------------------------------------------------------------------------

def life_cycle_cost(
    initial_cost: float,
    annual_operating_cost: float,
    annual_maintenance_cost: float,
    salvage_value: float,
    rate: float,
    life_years: int,
) -> dict:
    """Perform a Life-Cycle Cost Analysis (LCCA).

    Computes the total life-cycle cost (LCC) in present value terms,
    broken down by cost category.

    Parameters
    ----------
    initial_cost:
        One-time acquisition/deployment cost (year 0).
    annual_operating_cost:
        Recurring operating cost per year (cloud, power, bandwidth).
    annual_maintenance_cost:
        Recurring maintenance cost per year (calibration, replacements).
    salvage_value:
        Residual/salvage value at end of life (positive = asset value).
    rate:
        Discount rate per year (MARR).
    life_years:
        System service life in years.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``initial`` – initial cost (already at PV)
        - ``operating`` – PV of operating costs
        - ``maintenance`` – PV of maintenance costs
        - ``salvage`` – PV of salvage value (negative = benefit)
        - ``total`` – total LCC

    Examples
    --------
    >>> lcc = life_cycle_cost(50000, 5000, 2000, 8000, 0.08, 10)
    >>> round(lcc['total'], 2)
    93265.02
    """
    pv_operating = annuity_present_value(annual_operating_cost, rate, life_years)
    pv_maintenance = annuity_present_value(annual_maintenance_cost, rate, life_years)
    pv_salvage = present_value(salvage_value, rate, life_years)

    lcc = {
        "initial": initial_cost,
        "operating": pv_operating,
        "maintenance": pv_maintenance,
        "salvage": -pv_salvage,  # salvage reduces LCC
        "total": initial_cost + pv_operating + pv_maintenance - pv_salvage,
    }
    return lcc


# ---------------------------------------------------------------------------
# 9. Depreciation
# ---------------------------------------------------------------------------

def straight_line_depreciation(
    cost: float, salvage: float, life: int
) -> list:
    """Compute straight-line depreciation schedule.

    Parameters
    ----------
    cost:
        Initial asset cost.
    salvage:
        Salvage/residual value at end of life.
    life:
        Useful life in years.

    Returns
    -------
    list of dict
        One entry per year with keys ``year``, ``depreciation``,
        ``book_value``.

    Examples
    --------
    >>> schedule = straight_line_depreciation(50000, 5000, 5)
    >>> schedule[0]['depreciation']
    9000.0
    """
    annual_dep = (cost - salvage) / life
    schedule = []
    book_value = cost
    for year in range(1, life + 1):
        book_value -= annual_dep
        schedule.append(
            {
                "year": year,
                "depreciation": annual_dep,
                "book_value": round(book_value, 2),
            }
        )
    return schedule


# ---------------------------------------------------------------------------
# 10. Sensitivity Analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    base_cash_flows: Sequence[float],
    rate: float,
    parameter: str,
    variations: Sequence[float],
) -> list:
    """Perform a one-way sensitivity analysis on NPV.

    Varies a single parameter across *variations* while holding all others
    at their base values, and reports the resulting NPV.

    Parameters
    ----------
    base_cash_flows:
        Base-case cash flow sequence (period 0 first).
    rate:
        Base-case discount rate.
    parameter:
        Which parameter to vary: ``'rate'`` or ``'revenue'``.
        - ``'rate'``: varies the discount rate.
        - ``'revenue'``: scales all positive cash flows (inflows) by the
          variation factor.
    variations:
        Sequence of values to substitute for *parameter*.

    Returns
    -------
    list of dict
        Each entry contains ``parameter_value`` and ``npv``.

    Examples
    --------
    >>> cf = [-50000, 15000, 15000, 15000, 15000, 15000]
    >>> results = sensitivity_analysis(cf, 0.10, 'rate', [0.05, 0.10, 0.15])
    >>> [round(r['npv'], 2) for r in results]
    [14942.15, 6861.8, 282.33]
    """
    results = []
    for value in variations:
        if parameter == "rate":
            npv = net_present_value(value, base_cash_flows)
        elif parameter == "revenue":
            scaled = [
                cf * value if cf > 0 else cf for cf in base_cash_flows
            ]
            npv = net_present_value(rate, scaled)
        else:
            raise ValueError(f"Unsupported parameter '{parameter}'. Use 'rate' or 'revenue'.")
        results.append({"parameter_value": value, "npv": npv})
    return results


# ---------------------------------------------------------------------------
# 11. Wealth Creation Summary for Biometric Sensor System
# ---------------------------------------------------------------------------

def biometric_system_wealth_analysis(
    num_patients: int = 1000,
    monitoring_fee_per_patient_year: float = 600.0,
    hardware_cost_per_unit: float = 150.0,
    cloud_cost_per_patient_year: float = 120.0,
    maintenance_rate: float = 0.05,
    discount_rate: float = 0.10,
    life_years: int = 7,
) -> dict:
    """Comprehensive wealth creation analysis for a cloud-based biometric
    monitoring deployment.

    Models a healthcare IoT deployment where Arduino + AD8232 EKG sensor
    nodes stream data to the cloud for BPNN-based cardiac classification.
    Revenue is generated via a per-patient monitoring subscription.

    Parameters
    ----------
    num_patients:
        Number of patients monitored simultaneously.
    monitoring_fee_per_patient_year:
        Annual subscription revenue per patient (USD).
    hardware_cost_per_unit:
        One-time hardware cost per sensor node (USD).
    cloud_cost_per_patient_year:
        Annual cloud infrastructure cost per patient (USD).
    maintenance_rate:
        Fraction of initial hardware cost spent on maintenance each year.
    discount_rate:
        Minimum Attractive Rate of Return (MARR).
    life_years:
        Project planning horizon (years).

    Returns
    -------
    dict
        Comprehensive economic summary including NPV, IRR, payback, ROI,
        BCR, LCC breakdown, and depreciation schedule.
    """
    # --- Cash flow construction ---
    initial_investment = num_patients * hardware_cost_per_unit  # Year 0

    annual_revenue = num_patients * monitoring_fee_per_patient_year
    annual_op_cost = num_patients * cloud_cost_per_patient_year
    annual_maintenance = initial_investment * maintenance_rate
    annual_net = annual_revenue - annual_op_cost - annual_maintenance

    # Year-0 outflow, then annual net cash flows for life_years
    cash_flows = [-initial_investment] + [annual_net] * life_years

    # --- Core metrics ---
    npv = net_present_value(discount_rate, cash_flows)

    try:
        irr = internal_rate_of_return(cash_flows)
    except ValueError:
        irr = float("nan")

    pbp = simple_payback_period(cash_flows)
    dpbp = discounted_payback_period(cash_flows, discount_rate)

    total_net_benefit = annual_net * life_years - initial_investment
    roi = return_on_investment(total_net_benefit, initial_investment)

    # BCR
    benefits = [0] + [annual_revenue] * life_years
    costs_seq = [initial_investment] + [annual_op_cost + annual_maintenance] * life_years
    bcr = benefit_cost_ratio(benefits, costs_seq, discount_rate)

    # LCC
    lcc = life_cycle_cost(
        initial_investment,
        annual_op_cost,
        annual_maintenance,
        salvage_value=initial_investment * 0.10,  # 10 % residual
        rate=discount_rate,
        life_years=life_years,
    )

    # Depreciation (5-year straight-line for hardware)
    dep_life = min(5, life_years)
    dep_schedule = straight_line_depreciation(
        initial_investment, initial_investment * 0.10, dep_life
    )

    # Sensitivity on discount rate
    rate_sensitivity = sensitivity_analysis(
        cash_flows,
        discount_rate,
        "rate",
        [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    )

    return {
        "inputs": {
            "num_patients": num_patients,
            "monitoring_fee_per_patient_year_usd": monitoring_fee_per_patient_year,
            "hardware_cost_per_unit_usd": hardware_cost_per_unit,
            "cloud_cost_per_patient_year_usd": cloud_cost_per_patient_year,
            "maintenance_rate": maintenance_rate,
            "discount_rate_marr": discount_rate,
            "life_years": life_years,
        },
        "annual_summary": {
            "revenue_usd": annual_revenue,
            "operating_cost_usd": annual_op_cost,
            "maintenance_cost_usd": annual_maintenance,
            "net_cash_flow_usd": annual_net,
        },
        "wealth_metrics": {
            "npv_usd": round(npv, 2),
            "irr_percent": round(irr * 100, 2) if not math.isnan(irr) else "N/A",
            "simple_payback_years": round(pbp, 2),
            "discounted_payback_years": round(dpbp, 2),
            "roi_percent": round(roi, 2),
            "benefit_cost_ratio": round(bcr, 3),
        },
        "life_cycle_cost": {k: round(v, 2) for k, v in lcc.items()},
        "depreciation_schedule": dep_schedule,
        "npv_rate_sensitivity": [
            {"rate_percent": round(r["parameter_value"] * 100, 1),
             "npv_usd": round(r["npv"], 2)}
            for r in rate_sensitivity
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    """Run and display the engineering economics wealth analysis."""
    print("=" * 60)
    print("  Engineering Economics: Wealth Creation Analysis")
    print("  Cloud-based Biometric Sensor System")
    print("=" * 60)

    analysis = biometric_system_wealth_analysis()

    _print_section("System Inputs")
    inputs = analysis["inputs"]
    print(f"  Patients monitored          : {inputs['num_patients']:,}")
    print(f"  Monitoring fee / patient-yr : ${inputs['monitoring_fee_per_patient_year_usd']:,.2f}")
    print(f"  Hardware cost / node        : ${inputs['hardware_cost_per_unit_usd']:,.2f}")
    print(f"  Cloud cost / patient-yr     : ${inputs['cloud_cost_per_patient_year_usd']:,.2f}")
    print(f"  Maintenance rate            : {inputs['maintenance_rate']*100:.1f}% of hardware")
    print(f"  Discount rate (MARR)        : {inputs['discount_rate_marr']*100:.1f}%")
    print(f"  Planning horizon            : {inputs['life_years']} years")

    _print_section("Annual Cash Flow Summary")
    ann = analysis["annual_summary"]
    print(f"  Annual revenue              : ${ann['revenue_usd']:>12,.2f}")
    print(f"  Annual operating cost       : ${ann['operating_cost_usd']:>12,.2f}")
    print(f"  Annual maintenance cost     : ${ann['maintenance_cost_usd']:>12,.2f}")
    print(f"  Annual net cash flow        : ${ann['net_cash_flow_usd']:>12,.2f}")

    _print_section("Wealth Creation Metrics")
    wm = analysis["wealth_metrics"]
    npv_flag = "✓ VALUE CREATING" if wm["npv_usd"] > 0 else "✗ VALUE DESTROYING"
    print(f"  Net Present Value (NPV)     : ${wm['npv_usd']:>12,.2f}  {npv_flag}")
    irr_val = (
        f"{wm['irr_percent']:.2f}%" if isinstance(wm["irr_percent"], float)
        else wm["irr_percent"]
    )
    print(f"  Internal Rate of Return     :  {irr_val}")
    print(f"  Simple Payback Period       :  {wm['simple_payback_years']:.2f} years")
    print(f"  Discounted Payback Period   :  {wm['discounted_payback_years']:.2f} years")
    print(f"  Return on Investment (ROI)  :  {wm['roi_percent']:.2f}%")
    bcr_flag = "✓ ACCEPTABLE" if wm["benefit_cost_ratio"] > 1.0 else "✗ NOT JUSTIFIED"
    print(f"  Benefit-Cost Ratio (BCR)    :  {wm['benefit_cost_ratio']:.3f}  {bcr_flag}")

    _print_section("Life-Cycle Cost Analysis")
    lcc = analysis["life_cycle_cost"]
    print(f"  Initial cost (PV)           : ${lcc['initial']:>12,.2f}")
    print(f"  PV of operating costs       : ${lcc['operating']:>12,.2f}")
    print(f"  PV of maintenance costs     : ${lcc['maintenance']:>12,.2f}")
    print(f"  PV of salvage value         : ${lcc['salvage']:>12,.2f}")
    print(f"  Total Life-Cycle Cost       : ${lcc['total']:>12,.2f}")

    _print_section("Hardware Depreciation (Straight-Line)")
    print(f"  {'Year':<6} {'Depreciation':>14} {'Book Value':>12}")
    print(f"  {'-'*4:<6} {'-'*12:>14} {'-'*10:>12}")
    for row in analysis["depreciation_schedule"]:
        print(
            f"  {row['year']:<6} "
            f"${row['depreciation']:>12,.2f} "
            f"${row['book_value']:>11,.2f}"
        )

    _print_section("NPV Sensitivity to Discount Rate")
    print(f"  {'Rate (%)':>10} {'NPV (USD)':>14}")
    print(f"  {'-'*8:>10} {'-'*12:>14}")
    for row in analysis["npv_rate_sensitivity"]:
        print(f"  {row['rate_percent']:>9.1f}%  ${row['npv_usd']:>12,.2f}")

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
