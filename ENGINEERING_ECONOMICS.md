# Engineering Economics: Wealth Creation Analysis

## Cloud-based Biometric Sensor System — Economic Justification

This document provides a comprehensive engineering economics framework for
quantifying the wealth creation potential of the cloud-based biometric
monitoring system developed in this research.  It covers core time-value-of-money
concepts, investment appraisal metrics, life-cycle costing, and a worked
example tailored to the Arduino + AD8232 / BPNN deployment model described
in this repository.

The accompanying Python module [`engineering_economics.py`](engineering_economics.py)
implements all formulas below and can be executed directly:

```bash
python3 engineering_economics.py
```

---

## Table of Contents

1. [Economic Foundation](#1-economic-foundation)
2. [Time Value of Money](#2-time-value-of-money)
3. [Investment Appraisal Metrics](#3-investment-appraisal-metrics)
4. [Benefit-Cost Ratio](#4-benefit-cost-ratio)
5. [Break-Even Analysis](#5-break-even-analysis)
6. [Life-Cycle Cost Analysis](#6-life-cycle-cost-analysis)
7. [Depreciation](#7-depreciation)
8. [Sensitivity Analysis](#8-sensitivity-analysis)
9. [Worked Example: Biometric Monitoring Deployment](#9-worked-example-biometric-monitoring-deployment)
10. [Wealth Creation Summary](#10-wealth-creation-summary)

---

## 1. Economic Foundation

Engineering economics applies economic principles to evaluate the monetary
consequences of design and investment decisions over time.  The central premise
is **wealth creation**: a project creates value when its economic benefits
exceed its costs, measured in equivalent present-day dollars.

### Minimum Attractive Rate of Return (MARR)

The MARR $i^*$ is the lowest acceptable return an organisation will accept
on an investment, reflecting its cost of capital and risk profile:

$$\text{NPV} \geq 0 \iff \text{IRR} \geq i^*$$

Typical MARR values for healthcare technology projects: **8 – 15 %**.

---

## 2. Time Value of Money

### 2.1 Present Value

A cash flow $F$ received $n$ periods in the future has a present value:

$$P = \frac{F}{(1 + i)^n}$$

### 2.2 Future Value

A present amount $P$ grows to:

$$F = P \cdot (1 + i)^n$$

### 2.3 Annuity Present Value (P/A Factor)

A uniform annual payment $A$ for $n$ periods:

$$P = A \cdot \frac{1 - (1+i)^{-n}}{i}$$

### 2.4 Capital Recovery Factor (A/P)

The annual payment that repays principal $P$ over $n$ periods at rate $i$:

$$A = P \cdot \frac{i(1+i)^n}{(1+i)^n - 1}$$

---

## 3. Investment Appraisal Metrics

### 3.1 Net Present Value (NPV)

NPV discounts all future cash flows to the present at rate $i$:

$$\text{NPV} = \sum_{t=0}^{n} \frac{C_t}{(1+i)^t}$$

where $C_t$ is the net cash flow at period $t$ (negative for outflows).

| Decision rule | Interpretation |
|---|---|
| NPV > 0 | Project **creates wealth** — accept |
| NPV = 0 | Project earns exactly the MARR |
| NPV < 0 | Project **destroys wealth** — reject |

### 3.2 Internal Rate of Return (IRR)

The IRR is the discount rate $r$ that sets NPV to zero:

$$\sum_{t=0}^{n} \frac{C_t}{(1+r)^t} = 0$$

Solved iteratively via Newton-Raphson:

$$r_{k+1} = r_k - \frac{\text{NPV}(r_k)}{\text{NPV}'(r_k)}$$

where $\text{NPV}'(r) = \sum_{t=0}^{n} \frac{-t \cdot C_t}{(1+r)^{t+1}}$.

**Accept if IRR ≥ MARR.**

### 3.3 Simple Payback Period

The number of periods to recover the initial investment from cumulative
undiscounted net cash inflows:

$$\text{PBP} = t^* + \frac{|\text{Cumulative CF}_{t^*}|}{\text{CF}_{t^*+1}}$$

where $t^*$ is the last period with a negative cumulative balance.

### 3.4 Discounted Payback Period

Same as above, but applied to discounted cash flows
$\tilde{C}_t = C_t / (1+i)^t$:

$$\text{DPBP} = t^* + \frac{|\sum_{t=0}^{t^*} \tilde{C}_t|}{\tilde{C}_{t^*+1}}$$

### 3.5 Return on Investment (ROI)

$$\text{ROI} = \frac{\text{Net Benefit}}{\text{Total Investment}} \times 100\%$$

---

## 4. Benefit-Cost Ratio

$$\text{BCR} = \frac{\text{PV(Benefits)}}{\text{PV(Costs)}} = \frac{\sum_{t} B_t/(1+i)^t}{\sum_{t} C_t/(1+i)^t}$$

| BCR | Interpretation |
|---|---|
| BCR > 1 | Benefits exceed costs — project is **economically justified** |
| BCR = 1 | Break-even |
| BCR < 1 | Costs exceed benefits — project is **not justified** |

---

## 5. Break-Even Analysis

The break-even quantity $Q^*$ is where total revenue equals total cost:

$$Q^* = \frac{F}{P_u - V_u}$$

where:
- $F$ = fixed costs
- $P_u$ = price (revenue) per unit
- $V_u$ = variable cost per unit
- $(P_u - V_u)$ = **contribution margin** per unit

For the biometric system, "units" can represent individual sensor nodes
deployed or patients monitored per month.

---

## 6. Life-Cycle Cost Analysis

Life-Cycle Cost Analysis (LCCA) computes the total cost of ownership over
a project's service life in present-value terms:

$$\text{LCC} = C_{\text{initial}} + \text{PV}(C_{\text{operating}}) + \text{PV}(C_{\text{maintenance}}) - \text{PV}(S_{\text{salvage}})$$

where:
- $C_{\text{initial}}$ — one-time acquisition/deployment cost
- $C_{\text{operating}}$ — recurring cloud, power, and bandwidth costs
- $C_{\text{maintenance}}$ — sensor calibration, replacement parts
- $S_{\text{salvage}}$ — residual value of hardware at end of life

Recurring costs are computed using the annuity formula:

$$\text{PV}(C_{\text{annual}}) = C_{\text{annual}} \cdot \frac{1-(1+i)^{-n}}{i}$$

---

## 7. Depreciation

Depreciation allocates the cost of a capital asset over its useful life for
accounting and tax purposes.

### 7.1 Straight-Line (SL)

$$D_t = \frac{C - S}{n}$$

where $C$ is the initial cost, $S$ is salvage value, and $n$ is the asset life.

The book value at the end of year $t$:

$$B_t = C - t \cdot D_t$$

### 7.2 General MACRS Rates (IRS)

For 5-year class property (embedded hardware/sensors):

| Year | MACRS Rate |
|------|-----------|
| 1    | 20.00 %   |
| 2    | 32.00 %   |
| 3    | 19.20 %   |
| 4    | 11.52 %   |
| 5    | 11.52 %   |
| 6    |  5.76 %   |

---

## 8. Sensitivity Analysis

A one-way sensitivity analysis varies a single parameter while holding all
others constant, revealing which parameters most influence the NPV:

$$\text{NPV}(r) = \sum_{t=0}^{n} \frac{C_t}{(1+r)^t}$$

Common parameters to test:
- **Discount rate** $i$ (± 5 pp)
- **Annual revenue** (± 20 %)
- **Initial investment** (± 10 %)
- **Operating costs** (± 15 %)

A tornado chart ranks parameters by their impact spread on NPV.

---

## 9. Worked Example: Biometric Monitoring Deployment

### System Description

| Component | Specification |
|---|---|
| Sensor node | Arduino Rev3 ATmega328 + AD8232 |
| Inference engine | BPNN (4-class cardiac classifier, ~54k params) |
| Connectivity | Wi-Fi → Arduino Cloud |
| Revenue model | Per-patient annual monitoring subscription |
| Deployment scale | 1,000 patients |

### Cash Flow Assumptions

| Parameter | Value |
|---|---|
| Hardware cost per node | $150 |
| Annual monitoring fee per patient | $600 |
| Annual cloud cost per patient | $120 |
| Annual maintenance (5 % of hardware) | $7,500 |
| Salvage value (10 % of hardware) | $15,000 |
| Discount rate (MARR) | 10 % |
| Planning horizon | 7 years |

### Year-0 Investment

$$C_0 = 1{,}000 \times \$150 = \$150{,}000$$

### Annual Net Cash Flow

$$\text{Annual Revenue} = 1{,}000 \times \$600 = \$600{,}000$$

$$\text{Annual Cost} = \$120{,}000 + \$7{,}500 = \$127{,}500$$

$$C_{\text{annual}} = \$600{,}000 - \$127{,}500 = \$472{,}500$$

### Net Present Value

$$\text{NPV} = -150{,}000 + 472{,}500 \cdot \frac{1-(1.10)^{-7}}{0.10} \approx \$2{,}150{,}328$$

Because NPV > 0, the project **creates substantial wealth** at the 10 % MARR.

### Benefit-Cost Ratio

$$\text{BCR} = \frac{\text{PV(Revenue)}}{\text{PV(Total Cost)}} \approx 3.79$$

A BCR of 3.79 means that for every dollar invested, the project returns
approximately **$3.79 in discounted benefits**.

### Life-Cycle Cost Summary (PV, 7 years, 10 %)

| Cost Category | PV (USD) |
|---|---|
| Initial hardware | $150,000 |
| Cloud operating costs | $584,210 |
| Maintenance costs | $36,513 |
| Salvage value (benefit) | −$7,697 |
| **Total LCC** | **$763,026** |

### NPV Sensitivity to Discount Rate

| Discount Rate | NPV (USD) |
|---|---|
| 5 % | $2,584,061 |
| 8 % | $2,310,010 |
| 10 % | $2,150,328 |
| 12 % | $2,006,375 |
| 15 % | $1,815,798 |
| 20 % | $1,553,170 |

The project remains strongly positive across all tested discount rates,
demonstrating **robust wealth creation** even under pessimistic assumptions.

---

## 10. Wealth Creation Summary

The engineering economics analysis demonstrates that the cloud-based biometric
sensor platform is a sound investment with clear paths to wealth creation:

### Direct Financial Value

- **NPV ≈ $2.15 M** at MARR = 10 % over a 7-year horizon
- **IRR ≈ 315 %**, far exceeding typical healthcare technology hurdle rates
- **Simple payback period ≈ 0.32 years** (~4 months)
- **BCR ≈ 3.79** — each dollar invested returns $3.79 in benefits

### Indirect / Strategic Value

1. **Reduced hospital readmissions** — continuous remote monitoring enables
   early detection of cardiac events (Tachycardia, Bradycardia, Arrhythmia),
   potentially reducing costly emergency interventions.

2. **Scalability premium** — the BPNN classifier's small footprint (~54k
   parameters) enables TinyML edge deployment (TensorFlow Lite Micro),
   reducing cloud dependence and marginal cost at scale.

3. **Data network effects** — each additional patient improves classifier
   training data, increasing diagnostic accuracy and defensibility of the
   platform over time.

4. **Regulatory pathway value** — a validated, documented BPNN with
   mathematical formulations (see [`BPNN_MATHEMATICAL_FORMULATIONS.md`](BPNN_MATHEMATICAL_FORMULATIONS.md))
   accelerates FDA 510(k) or CE-Mark submissions, shortening time-to-market.

### Sensitivity Robustness

The project remains NPV-positive across all discount rates tested (5–20 %),
confirming that wealth creation is not contingent on optimistic financing
assumptions.  Even if annual revenue were reduced by 50 %, the project would
still reach payback within the 7-year horizon.

---

## References

- Blank, L. & Tarquin, A. (2017). *Engineering Economy*, 8th ed. McGraw-Hill.
- White, J. A. et al. (2015). *Principles of Engineering Economic Analysis*, 6th ed. Wiley.
- Park, C. S. (2016). *Contemporary Engineering Economics*, 6th ed. Pearson.
- IRS Publication 946 — *How to Depreciate Property* (MACRS tables).
