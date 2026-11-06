import os
import pytest
from app.analyzer import analyze_financial_data

def test_analyzer_execution():
    # Run analysis pointing test artifacts to a dedicated test directory
    facts = analyze_financial_data(output_dir="outputs/test_run")

    # 1. Verify structure
    assert "dataset_profile" in facts
    assert "financial_statistics_inr" in facts
    assert "outlier_analysis" in facts
    assert "chart_artifacts" in facts

    # 2. Verify strict deterministic math
    stats = facts["financial_statistics_inr"]
    assert stats["valid_transaction_count"] == 314
    assert stats["total_amount"] == 3826679.43
    assert stats["mean_amount"] == 12186.88

    # 3. Verify test artifacts exist
    output_dir = facts["output_directory"]
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "financial_facts.json"))
    assert os.path.exists(os.path.join(output_dir, "cleaned_transactions.csv"))
    assert os.path.exists(os.path.join(output_dir, "amount_distribution.png"))