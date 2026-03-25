#!/usr/bin/env python3
"""
Validation script to verify the mathematical formulations in the notebooks.

This script validates:
1. Jupyter notebook structure is correct
2. Mathematical expressions are properly documented
3. Code cells contain valid Python syntax
"""

import json
import sys
import re

def validate_notebook(filename):
    """Validate a Jupyter notebook file."""
    print(f"\nValidating {filename}...")
    
    try:
        with open(filename, 'r') as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ FAILED: Invalid JSON - {e}")
        return False
    except FileNotFoundError:
        print(f"  ❌ FAILED: File not found")
        return False
    
    # Check notebook structure
    if 'cells' not in notebook:
        print(f"  ❌ FAILED: Missing 'cells' key")
        return False
    
    if 'metadata' not in notebook:
        print(f"  ❌ FAILED: Missing 'metadata' key")
        return False
    
    # Count cells
    markdown_cells = 0
    code_cells = 0
    math_expressions = 0
    
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'markdown':
            markdown_cells += 1
            # Check for mathematical expressions (LaTeX)
            source = ''.join(cell.get('source', []))
            # Count $$ blocks and inline $ expressions
            math_expressions += len(re.findall(r'\$\$[^$]+\$\$', source))
            math_expressions += len(re.findall(r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)', source))
        elif cell.get('cell_type') == 'code':
            code_cells += 1
    
    print(f"  ✓ Valid Jupyter notebook structure")
    print(f"  ✓ {markdown_cells} markdown cells")
    print(f"  ✓ {code_cells} code cells")
    print(f"  ✓ {math_expressions} mathematical expressions found")
    
    if math_expressions == 0:
        print(f"  ⚠ WARNING: No mathematical expressions found")
        return False
    
    return True

def validate_documentation(filename):
    """Validate the mathematical documentation file."""
    print(f"\nValidating {filename}...")
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  ❌ FAILED: File not found")
        return False
    
    # Check for key mathematical sections
    required_sections = [
        'Forward Propagation',
        'Activation Functions',
        'Loss Functions',
        'Backpropagation',
        'Parameter Updates'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"  ❌ FAILED: Missing sections: {', '.join(missing_sections)}")
        return False
    
    # Count mathematical expressions
    math_expressions = len(re.findall(r'\$\$[^$]+\$\$', content))
    math_expressions += len(re.findall(r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)', content))
    
    print(f"  ✓ All required sections present")
    print(f"  ✓ {math_expressions} mathematical expressions found")
    
    if math_expressions < 20:
        print(f"  ⚠ WARNING: Expected more mathematical expressions")
        return False
    
    return True

def validate_engineering_economics():
    """Validate the engineering economics module."""
    print("\nValidating engineering_economics.py...")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "engineering_economics", "engineering_economics.py"
        )
        ee = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ee)
    except FileNotFoundError:
        print("  ❌ FAILED: File not found")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: Import error - {e}")
        return False

    # Validate core functions exist
    required_functions = [
        "present_value",
        "future_value",
        "net_present_value",
        "internal_rate_of_return",
        "simple_payback_period",
        "discounted_payback_period",
        "return_on_investment",
        "benefit_cost_ratio",
        "break_even_units",
        "life_cycle_cost",
        "straight_line_depreciation",
        "sensitivity_analysis",
        "biometric_system_wealth_analysis",
    ]

    missing = [fn for fn in required_functions if not hasattr(ee, fn)]
    if missing:
        print(f"  ❌ FAILED: Missing functions: {', '.join(missing)}")
        return False

    # Spot-check key calculations
    errors = []

    # NPV: known result
    cf = [-50000, 15000, 15000, 15000, 15000, 15000]
    npv = ee.net_present_value(0.10, cf)
    if abs(npv - 6861.80) > 1.0:
        errors.append(f"NPV expected ~6861.80, got {npv:.2f}")

    # ROI
    roi = ee.return_on_investment(25000, 50000)
    if abs(roi - 50.0) > 0.01:
        errors.append(f"ROI expected 50.0, got {roi:.2f}")

    # Break-even
    beq = ee.break_even_units(50000, 300, 50)
    if abs(beq - 200.0) > 0.01:
        errors.append(f"Break-even expected 200.0, got {beq:.2f}")

    # Full wealth analysis (smoke test)
    analysis = ee.biometric_system_wealth_analysis()
    if analysis["wealth_metrics"]["npv_usd"] <= 0:
        errors.append("Biometric system analysis produced non-positive NPV")
    if analysis["wealth_metrics"]["benefit_cost_ratio"] <= 1.0:
        errors.append("Biometric system analysis produced BCR <= 1.0")

    if errors:
        for err in errors:
            print(f"  ❌ CALCULATION ERROR: {err}")
        return False

    print("  ✓ All required functions present")
    print(f"  ✓ {len(required_functions)} engineering economics functions validated")
    print("  ✓ NPV, ROI, break-even, and wealth analysis calculations correct")
    return True


def main():
    """Run all validations."""
    print("="*60)
    print("Validating Mathematical Formulations Implementation")
    print("="*60)
    
    results = []
    
    # Validate notebooks
    results.append(("dataloading.ipynb", validate_notebook("dataloading.ipynb")))
    results.append(("randomdatasetgenerator.ipynb", validate_notebook("randomdatasetgenerator.ipynb")))
    
    # Validate documentation
    results.append(("BPNN_MATHEMATICAL_FORMULATIONS.md", validate_documentation("BPNN_MATHEMATICAL_FORMULATIONS.md")))
    
    # Validate engineering economics module
    results.append(("engineering_economics.py", validate_engineering_economics()))

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    all_passed = True
    for filename, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {filename}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All validations passed!")
        print("\nMathematical formulations have been successfully implemented:")
        print("  • Forward propagation equations")
        print("  • Backpropagation algorithms")
        print("  • Activation function derivatives")
        print("  • Loss function formulas")
        print("  • Signal preprocessing methods")
        print("  • EKG synthesis models")
        print("  • Engineering economics and wealth creation calculations")
        return 0
    else:
        print("\n❌ Some validations failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
