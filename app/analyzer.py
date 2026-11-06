import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_COLUMNS = [
    "transaction_id",
    "reference_number",
    "transaction_date",
    "transaction_amount",
    "currency",
    "transaction_type",
    "category",
    "status",
    "account_id",
    "account_type",
    "channel",
    "customer_region",
]


def analyze_financial_data(
    csv_path: str = "data/fintech_transactions_training_dataset.csv",
    output_dir: str = "outputs",
) -> dict:
    """
    Deterministic Financial Data Processing & Analysis.
    Routes production runs to outputs/runs/run_YYYY-MM-DD_HH-MM-SS/
    and test runs to outputs/test_run/run_YYYY-MM-DD_HH-MM-SS/.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV file not found at: {csv_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Route production runs into outputs/runs/, and test runs into their specified folder with a timestamp
    if output_dir == "outputs":
        output_dir = os.path.join("outputs", "runs", f"run_{timestamp}")
    else:
        output_dir = os.path.join(output_dir, f"run_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ingestion & Schema Validation

    raw_df = pd.read_csv(csv_path)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing columns: {missing_cols}")

    total_raw_rows = int(len(raw_df))
    total_columns = int(len(raw_df.columns))

    # 2. Data Profiling & Quality Checks on Raw Data

    missing_values_by_col = {
        col: int(raw_df[col].isna().sum()) for col in EXPECTED_COLUMNS
    }
    total_missing_cells = int(sum(missing_values_by_col.values()))

    exact_duplicate_mask = raw_df.duplicated(keep="first")
    exact_duplicate_count = int(exact_duplicate_mask.sum())
    duplicate_transaction_ids = (
        raw_df.loc[exact_duplicate_mask, "transaction_id"].dropna().astype(str).tolist()
    )

    raw_parsed_dates = pd.to_datetime(
        raw_df["transaction_date"], format="%Y-%m-%d", errors="coerce"
    )
    invalid_date_mask = raw_parsed_dates.isna() & raw_df["transaction_date"].notna()
    invalid_date_records = [
        {
            "transaction_id": str(row["transaction_id"]),
            "raw_date": str(row["transaction_date"]),
        }
        for _, row in raw_df.loc[invalid_date_mask].iterrows()
    ]

    raw_parsed_amounts = pd.to_numeric(raw_df["transaction_amount"], errors="coerce")
    invalid_amount_mask = (
        raw_parsed_amounts.isna() & raw_df["transaction_amount"].notna()
    )
    invalid_amount_records = [
        {
            "transaction_id": str(row["transaction_id"]),
            "raw_amount": str(row["transaction_amount"]),
        }
        for _, row in raw_df.loc[invalid_amount_mask].iterrows()
    ]

    # 3. Create Cleaned Analysis DataFrame

    clean_df = raw_df.drop_duplicates(keep="first").copy()
    clean_df["parsed_date"] = pd.to_datetime(
        clean_df["transaction_date"], format="%Y-%m-%d", errors="coerce"
    )
    clean_df["parsed_amount"] = pd.to_numeric(
        clean_df["transaction_amount"], errors="coerce"
    )

    cleaned_csv_path = os.path.join(output_dir, "cleaned_transactions.csv").replace("\\", "/")
    clean_df.to_csv(cleaned_csv_path, index=False)

    valid_amount_df = clean_df[
        clean_df["parsed_amount"].notna() & (clean_df["parsed_amount"] > 0)
    ].copy()
    amounts = valid_amount_df["parsed_amount"]

    currencies = valid_amount_df["currency"].dropna().unique().tolist()
    primary_currency = currencies[0] if len(currencies) == 1 else ", ".join(currencies)

    # 4. Deterministic Financial Statistics

    valid_txn_count = int(len(amounts))
    total_amount = round(float(amounts.sum()), 2)
    mean_amount = round(float(amounts.mean()), 2)
    median_amount = round(float(amounts.median()), 2)
    std_amount = round(float(amounts.std()), 2)
    min_amount = round(float(amounts.min()), 2)
    max_amount = round(float(amounts.max()), 2)

    q1_25 = round(float(amounts.quantile(0.25)), 2)
    q3_75 = round(float(amounts.quantile(0.75)), 2)
    p90 = round(float(amounts.quantile(0.90)), 2)
    p95 = round(float(amounts.quantile(0.95)), 2)
    p99 = round(float(amounts.quantile(0.99)), 2)
    iqr_value = round(float(q3_75 - q1_25), 2)

    valid_dates = clean_df["parsed_date"].dropna()
    start_date = str(valid_dates.min().date()) if not valid_dates.empty else "N/A"
    end_date = str(valid_dates.max().date()) if not valid_dates.empty else "N/A"

    # 5. Outlier Identification (1.5 * IQR Rule)

    lower_fence = round(float(q1_25 - 1.5 * iqr_value), 2)
    upper_fence = round(float(q3_75 + 1.5 * iqr_value), 2)

    outlier_df = valid_amount_df[
        (valid_amount_df["parsed_amount"] < lower_fence)
        | (valid_amount_df["parsed_amount"] > upper_fence)
    ].sort_values(by="parsed_amount", ascending=False)

    outlier_records = [
        {
            "transaction_id": str(row["transaction_id"]),
            "transaction_date": str(row["transaction_date"]),
            "amount_inr": round(float(row["parsed_amount"]), 2),
            "category": str(row["category"]) if pd.notna(row["category"]) else "Missing",
            "status": str(row["status"]) if pd.notna(row["status"]) else "Missing",
            "channel": str(row["channel"]),
            "customer_region": str(row["customer_region"]),
            "flag_reason": "Exceeds Upper IQR Fence (Review Candidate)",
        }
        for _, row in outlier_df.iterrows()
    ]
    extreme_outliers_count = int((outlier_df["parsed_amount"] >= 100000).sum())

    # 6. Group-By Summaries

    def build_group_summary(df_subset: pd.DataFrame, group_col: str) -> dict:
        temp = df_subset.copy()
        temp[group_col] = temp[group_col].fillna("Missing")
        grouped = temp.groupby(group_col)["parsed_amount"].agg(["count", "sum", "mean", "median"])
        result = {}
        for idx, row in grouped.sort_values(by="sum", ascending=False).iterrows():
            result[str(idx)] = {
                "transaction_count": int(row["count"]),
                "total_amount_inr": round(float(row["sum"]), 2),
                "mean_amount_inr": round(float(row["mean"]), 2),
                "median_amount_inr": round(float(row["median"]), 2),
            }
        return result

    category_summary = build_group_summary(valid_amount_df, "category")
    status_summary = build_group_summary(valid_amount_df, "status")
    channel_summary = build_group_summary(valid_amount_df, "channel")
    region_summary = build_group_summary(valid_amount_df, "customer_region")
    type_summary = build_group_summary(valid_amount_df, "transaction_type")
    account_type_summary = build_group_summary(valid_amount_df, "account_type")

    valid_date_amount_df = valid_amount_df[valid_amount_df["parsed_date"].notna()].copy()
    valid_date_amount_df["year_month"] = valid_date_amount_df["parsed_date"].dt.strftime("%Y-%m")
    monthly_grouped = (
        valid_date_amount_df.groupby("year_month")["parsed_amount"]
        .agg(["count", "sum", "mean"])
        .sort_index()
    )
    monthly_summary = {
        str(month): {
            "transaction_count": int(row["count"]),
            "total_amount_inr": round(float(row["sum"]), 2),
            "mean_amount_inr": round(float(row["mean"]), 2),
        }
        for month, row in monthly_grouped.iterrows()
    }

    # 7. Visualizations (Saved inside the timestamped run folder)

    chart_paths = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    normal_amounts = amounts[amounts <= upper_fence]
    axes[0].hist(normal_amounts, bins=20, color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0].set_title("Regular Transaction Amounts (<= Upper IQR Fence)")
    axes[0].set_xlabel("Amount (INR)")
    axes[0].set_ylabel("Transaction Count")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(amounts, bins=25, color="#d62728", edgecolor="black", alpha=0.85, log=True)
    axes[1].set_title("All Transaction Amounts (Log-Scale Frequency)")
    axes[1].set_xlabel("Amount (INR)")
    axes[1].set_ylabel("Log Frequency")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Fintech Transaction Amount Distribution (INR)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "amount_distribution.png").replace("\\", "/")
    plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths["amount_distribution"] = chart1_path

    fig, ax1 = plt.subplots(figsize=(10, 5))
    months = list(monthly_summary.keys())
    monthly_totals = [monthly_summary[m]["total_amount_inr"] for m in months]
    monthly_counts = [monthly_summary[m]["transaction_count"] for m in months]

    ax1.bar(months, monthly_totals, color="#2a9d8f", alpha=0.75, label="Total Value (INR)")
    ax1.set_xlabel("Month (2025)")
    ax1.set_ylabel("Total Transaction Value (INR)", color="#2a9d8f", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(months, monthly_counts, color="#e76f51", marker="o", linewidth=2.5, label="Transaction Count")
    ax2.set_ylabel("Transaction Volume (Count)", color="#e76f51", fontweight="bold")

    plt.title("Monthly Transaction Value (INR) & Volume Over Time (2025)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "volume_over_time.png").replace("\\", "/")
    plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths["volume_over_time"] = chart2_path

    plt.figure(figsize=(9, 4.8))
    quality_labels = [
        "Missing: category",
        "Missing: account_id",
        "Missing: reference_no",
        "Missing: amount",
        "Missing: status",
        "Invalid Date Format",
        "Invalid Amount Text",
        "Exact Duplicate Rows",
    ]
    quality_counts = [
        missing_values_by_col["category"],
        missing_values_by_col["account_id"],
        missing_values_by_col["reference_number"],
        missing_values_by_col["transaction_amount"],
        missing_values_by_col["status"],
        len(invalid_date_records),
        len(invalid_amount_records),
        exact_duplicate_count,
    ]
    bars = plt.barh(quality_labels, quality_counts, color="#f4a261", edgecolor="black")
    plt.xlabel("Record Count")
    plt.title("Data Quality Findings: Missing Values, Invalid Formats & Duplicates", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontweight="bold")
    plt.xlim(0, max(quality_counts) + 1.5)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "missing_values.png").replace("\\", "/")
    plt.savefig(chart3_path, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths["missing_values"] = chart3_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cat_names = list(category_summary.keys())
    cat_totals = [category_summary[c]["total_amount_inr"] for c in cat_names]
    axes[0].barh(cat_names[::-1], cat_totals[::-1], color="#457b9d", edgecolor="black")
    axes[0].set_title("Total Value by Business Category (INR)")
    axes[0].set_xlabel("Total Amount (INR)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    chan_names = list(channel_summary.keys())
    chan_counts = [channel_summary[c]["transaction_count"] for c in chan_names]
    axes[1].bar(chan_names, chan_counts, color="#6d597a", edgecolor="black")
    axes[1].set_title("Transaction Count by Channel")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Category Value & Channel Usage Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "category_channel_breakdown.png").replace("\\", "/")
    plt.savefig(chart4_path, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths["category_channel_breakdown"] = chart4_path

    # 8. Build Final Verified JSON Payload
    facts = {
        "run_timestamp": timestamp,
        "output_directory": output_dir,
        "dataset_profile": {
            "source_file": csv_path,
            "raw_row_count": total_raw_rows,
            "column_count": total_columns,
            "columns": EXPECTED_COLUMNS,
            "deduplicated_row_count": int(len(clean_df)),
            "valid_amount_row_count": valid_txn_count,
            "currency": primary_currency,
            "date_coverage": {"start_date": start_date, "end_date": end_date},
        },
        "data_quality_summary": {
            "exact_duplicate_rows_count": exact_duplicate_count,
            "duplicate_transaction_ids": duplicate_transaction_ids,
            "missing_values_by_column": missing_values_by_col,
            "total_missing_cells": total_missing_cells,
            "invalid_date_count": len(invalid_date_records),
            "invalid_amount_count": len(invalid_amount_records),
        },
        "financial_statistics_inr": {
            "valid_transaction_count": valid_txn_count,
            "total_amount": total_amount,
            "mean_amount": mean_amount,
            "median_amount": median_amount,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "percentiles": {
                "p25_q1": q1_25,
                "p50_median": median_amount,
                "p75_q3": q3_75,
                "iqr": iqr_value,
            },
        },
        "outlier_analysis": {
            "method": "1.5 * IQR Rule (Review Candidates)",
            "lower_fence_inr": lower_fence,
            "upper_fence_inr": upper_fence,
            "total_outlier_candidates": len(outlier_records),
            "outlier_review_table": outlier_records,
        },
        "segmented_summaries": {
            "by_month": monthly_summary,
            "by_category": category_summary,
            "by_status": status_summary,
            "by_channel": channel_summary,
            "by_customer_region": region_summary,
        },
        "chart_artifacts": chart_paths,
    }

    json_output_path = os.path.join(output_dir, "financial_facts.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)

    print(f"Analysis Complete: Output neatly saved to folder -> {output_dir}/")
    return facts


if __name__ == "__main__":
    analyze_financial_data()