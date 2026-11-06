# Fintech Transactions Dataset Guide

## Purpose

This synthetic, non-confidential dataset supports the Automated Financial Data Analysis, GenAI Report, and RAG Chatbot proof of concept. Each row represents one financial transaction.

The dataset is designed for exploratory data analysis, data-quality checks, financial summaries, chart generation, structured JSON output, and a short business report.

## Dataset Structure

The CSV contains 327 rows and 12 columns covering transactions from 2025. Amounts are recorded in INR.

| Column | Description |
| --- | --- |
| `transaction_id` | Identifier assigned to the transaction |
| `reference_number` | External or business reference for the transaction |
| `transaction_date` | Date on which the transaction was recorded |
| `transaction_amount` | Transaction value in the stated currency |
| `currency` | Currency code for the transaction amount |
| `transaction_type` | Indicates whether the transaction is a debit or credit |
| `category` | Business category such as fund transfer or bill payment |
| `status` | Processing status such as completed, pending, failed, or reversed |
| `account_id` | Identifier of the related customer account |
| `account_type` | Savings, current, or wallet account |
| `channel` | Channel used to perform the transaction |
| `customer_region` | Broad geographical region associated with the customer |

## Feature Categories and Expected Values

| Column | Feature category | Expected values or format | EDA use |
| --- | --- | --- | --- |
| `transaction_id` | Identifier | Text such as `TXN202500001`; expected to identify a transaction | Check uniqueness and possible duplicates; do not calculate averages or totals |
| `reference_number` | Identifier | Text such as `REF202500001` | Check missing values and repeated references |
| `transaction_date` | Temporal | Expected format `YYYY-MM-DD` | Parse as a date and use for date coverage and monthly trends |
| `transaction_amount` | Numerical continuous | Positive INR amount with two decimal places | Calculate totals, averages, percentiles, distributions, and potential outliers |
| `currency` | Categorical nominal | `INR` | Confirm currency consistency before aggregating amounts |
| `transaction_type` | Categorical nominal | `Debit`, `Credit` | Compare transaction counts and values by financial direction |
| `category` | Categorical nominal | `Merchant Purchase`, `Fund Transfer`, `Bill Payment`, `Cash Withdrawal`, `Subscription`, `Refund` | Compare transaction volume and value by business category |
| `status` | Categorical nominal | `Completed`, `Pending`, `Failed`, `Reversed` | Review processing outcomes and status distribution |
| `account_id` | Identifier and grouping feature | Text such as `ACC1001`; the same account can appear in several rows | Group transactions by account and check missing identifiers |
| `account_type` | Categorical nominal | `Savings`, `Current`, `Wallet` | Compare activity across account types |
| `channel` | Categorical nominal | `Mobile`, `Web`, `API`, `POS`, `ATM`, `Auto Debit` | Compare usage and transaction value by channel |
| `customer_region` | Categorical nominal | `North`, `South`, `East`, `West`, `Central` | Compare regional transaction activity |

### Feature Group Summary

- **Identifiers:** `transaction_id`, `reference_number`, `account_id`
- **Temporal feature:** `transaction_date`
- **Numerical feature:** `transaction_amount`
- **Transaction categories:** `currency`, `transaction_type`, `category`, `status`
- **Customer and channel categories:** `account_type`, `channel`, `customer_region`

Categorical columns should generally be analysed with value counts, percentages, group-by summaries, and bar charts. Identifier columns should mainly be used for uniqueness, missing-value, duplicate, and grouping checks.

## Important Training Note

The CSV intentionally contains a small number of missing values, invalid dates, invalid amounts, exact duplicate rows, and unusually high transaction amounts. These records are included for EDA and data-cleaning practice. Students should detect and report them instead of silently deleting or correcting them.

## Recommended EDA Process

1. Load the CSV into a Pandas DataFrame.
2. Review its shape, columns, sample rows, and inferred data types.
3. Convert transaction dates and amounts using controlled error handling.
4. Identify missing values, invalid formats, and duplicate records.
5. Calculate count, total, mean, median, minimum, maximum, and percentiles for valid amounts.
6. Analyse transaction volume and value by month, category, status, channel, and region.
7. Identify potential outliers using IQR or an agreed business threshold.
8. Create clear charts and store verified findings in structured JSON.

## Analysis Rules

- Keep the original CSV unchanged and perform cleaning in a separate DataFrame.
- Record how missing, invalid, and duplicate records are handled.
- Do not treat invalid or missing amounts as zero.
- Label outliers as review candidates, not confirmed errors or fraud.
- Use deterministic Python calculations for all statistics.
- The LLM may explain verified JSON results but must not recalculate financial figures.

## Expected Student Outputs

- Dataset profile
- Data-quality summary
- Cleaned analysis DataFrame
- Financial statistics
- Monthly and category summaries
- Potential outlier table
- Four or five relevant charts
- Structured analysis JSON
- Short business-readable analysis report

