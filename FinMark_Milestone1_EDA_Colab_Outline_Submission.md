# MO-IT162 Milestone 1: ML Solution EDA
## Refined Colab Outline and Analysis Guide

> **Purpose:** This guide supports the FinMark Corporation Milestone 1 Exploratory Data Analysis (EDA) using the finalized Week 3 preprocessed datasets.
>
> Milestone 1 focuses on **statistical exploration, distributions, grouped comparisons, relationships, anomalies, and interpretation** using Python and Pandas.
>
> **Scope boundary:** Polished charts, dashboard design, and visual storytelling are reserved for **Milestone 2: ML Solution DataViz**. Milestone 1 will rely mainly on statistical outputs and summary tables.
>
> **Notebook convention:** Explanations may be written as concise comments beside the code they explain.

---

# 1. Project Context and Problem Identification

## 1.1 FinMark Business Problem

FinMark Corporation collects customer information from three sources: customer demographics, customer transactions, and social media interactions.

Before the data can support future machine learning activities, the team must understand its structure, statistical characteristics, distributions, patterns, relationships, and remaining limitations.

Milestone 1 therefore uses Exploratory Data Analysis (EDA) to examine the finalized Week 3 preprocessed datasets and determine what the available evidence shows.

## 1.2 Milestone 1 Objectives

The EDA will:

1. review the structure of the finalized Week 3 datasets;
2. summarize the preprocessing decisions performed during Weeks 2–3;
3. classify numerical, categorical, identifier, and temporal variables;
4. calculate measures of central tendency;
5. calculate measures of variability;
6. review quartiles, distributions, and potential outliers;
7. examine categorical frequencies and possible imbalance;
8. compare meaningful groups;
9. investigate relationships and trends where appropriate;
10. identify issues that may affect future machine learning activities; and
11. summarize evidence-supported findings, limitations, and considerations.

## 1.3 Scope

Milestone 1 focuses on **EDA and machine learning preparation**.

It does **not**:

- train or evaluate a machine learning model;
- produce the final Milestone 2 visualizations or dashboard;
- establish causal relationships from associations; or
- make recommendations unsupported by the available FinMark data.

---

# 2. Import Libraries

```python
# Pandas is used for loading, inspecting, grouping, and summarizing the datasets.
# NumPy supports numerical calculations when needed.

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
```

---

# 3. Load the Finalized Week 3 Preprocessed Datasets

Use the cleaned files produced and validated during Week 3.

```python
# Update the path prefix only if the files are stored in another Colab folder.

demographics_path = "/content/customer_demographics_cleaned.csv"
transactions_path = "/content/customer_transactions_cleaned.csv"
social_path = "/content/social_media_interactions_cleaned.csv"

demographics = pd.read_csv(demographics_path)
transactions = pd.read_csv(transactions_path)
social = pd.read_csv(social_path)

print("Demographics shape:", demographics.shape)
print("Transactions shape:", transactions.shape)
print("Social media shape:", social.shape)
```

Expected Week 3 row counts:

| Dataset | Finalized Rows |
|---|---:|
| Customer Demographics | 2,378 |
| Customer Transactions | 2,420 |
| Social Media Interactions | 2,411 |

If the loaded row counts differ, verify that the correct Week 3 files were uploaded before continuing.

---

# 4. Dataset Overview and Structure

## 4.1 Customer Demographics

```python
display(demographics.head())
print("Shape:", demographics.shape)
print("Columns:", demographics.columns.tolist())
demographics.info()

print("\nUnique values per column:")
display(demographics.nunique())
```

## 4.2 Customer Transactions

```python
display(transactions.head())
print("Shape:", transactions.shape)
print("Columns:", transactions.columns.tolist())
transactions.info()

print("\nUnique values per column:")
display(transactions.nunique())
```

## 4.3 Social Media Interactions

```python
display(social.head())
print("Shape:", social.shape)
print("Columns:", social.columns.tolist())
social.info()

print("\nUnique values per column:")
display(social.nunique())
```

---

# 5. Variable Classification

The Week 3 cleaned datasets use `snake_case` column names.

| Dataset | Variable | Statistical Type | Analytical Role |
|---|---|---|---|
| Demographics | `customer_id` | Identifier | Relationship key |
| Demographics | `age` | Quantitative | Ratio |
| Demographics | `gender` | Qualitative | Nominal |
| Demographics | `location` | Qualitative | Nominal / geographic |
| Demographics | `income_level` | Qualitative | Ordinal if the ordering is validated |
| Demographics | `signup_date` | Temporal | Date |
| Transactions | `customer_id` | Identifier | Relationship key |
| Transactions | `transaction_id` | Identifier | Record key |
| Transactions | `transaction_date` | Temporal | Date |
| Transactions | `amount` | Quantitative | Ratio |
| Transactions | `product_category` | Qualitative | Nominal |
| Transactions | `payment_method` | Qualitative | Nominal |
| Social Media | `customer_id` | Identifier | Relationship key |
| Social Media | `interaction_id` | Identifier | Record key |
| Social Media | `interaction_date` | Temporal | Date |
| Social Media | `platform` | Qualitative | Nominal |
| Social Media | `interaction_type` | Qualitative | Nominal |
| Social Media | `sentiment` | Qualitative | Ordered only if the scale is explicitly treated as ordinal |

```python
print("Demographics dtypes:")
display(demographics.dtypes)

print("\nTransactions dtypes:")
display(transactions.dtypes)

print("\nSocial media dtypes:")
display(social.dtypes)
```

**Important:** An identifier may be stored as a number but should not be treated as a quantitative measurement. For example, calculating the mean of `customer_id` would have no analytical meaning.

---

# 6. Preprocessing Review and Post-Cleaning Validation

This section confirms that the Week 3 outputs remain suitable for EDA.

Use the reporting pattern:

**Issue → Treatment → Reason → Validation**

## 6.1 Missing Values

```python
print("Demographics missing values:")
display(demographics.isna().sum())

print("\nTransactions missing values:")
display(transactions.isna().sum())

print("\nSocial media missing values:")
display(social.isna().sum())
```

The finalized Week 3 exports were previously validated with zero missing values. Any unexpected missing values should be investigated before continuing.

## 6.2 Exact Duplicate Rows

```python
print("Demographics exact duplicates:", demographics.duplicated().sum())
print("Transactions exact duplicates:", transactions.duplicated().sum())
print("Social media exact duplicates:", social.duplicated().sum())
```

## 6.3 Record Identifier Integrity

```python
print(
    "Duplicated demographic customer IDs:",
    demographics["customer_id"].duplicated().sum()
)

print(
    "Duplicated transaction IDs:",
    transactions["transaction_id"].duplicated().sum()
)

print(
    "Duplicated interaction IDs:",
    social["interaction_id"].duplicated().sum()
)
```

Repeated `customer_id` values in the transaction and social datasets are expected because one customer may have multiple activities. `transaction_id` and `interaction_id`, however, should identify individual records.

## 6.4 Date Validation

```python
date_columns = [
    (demographics, "signup_date"),
    (transactions, "transaction_date"),
    (social, "interaction_date")
]

for df, column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")
    print(f"{column} unparseable/missing dates:", df[column].isna().sum())
```

## 6.5 Post-Cleaning Cross-Dataset Coverage

Week 3 documented that independent cleaning created some activity records whose demographic record was no longer present in the cleaned demographics dataset.

```python
demographic_ids = set(demographics["customer_id"])
transaction_ids = set(transactions["customer_id"])
social_ids = set(social["customer_id"])

print(
    "Transaction customer IDs without clean demographics:",
    len(transaction_ids - demographic_ids)
)

print(
    "Social customer IDs without clean demographics:",
    len(social_ids - demographic_ids)
)
```

Treat this as an **analytical coverage limitation**, not automatically as a source-integrity error.

---

# 7. Descriptive Statistics

## 7.1 General Numerical Summary

```python
print("Customer Age Summary")
display(demographics["age"].describe())

print("\nTransaction Amount Summary")
display(transactions["amount"].describe())
```

Interpret:

- `count` as the number of observations used;
- `mean` as the average;
- `std` as the standard deviation;
- `min` and `max` as the observed extremes; and
- `25%`, `50%`, and `75%` as quartile positions.

Do not report the table without explaining what the values mean.

---

# 8. Measures of Central Tendency

## 8.1 Customer Age

```python
age = demographics["age"]

age_mean = age.mean()
age_median = age.median()
age_mode = age.mode()

print(f"Mean Age: {age_mean:.2f}")
print(f"Median Age: {age_median:.2f}")
print("Mode Age(s):", age_mode.tolist())
```

Review whether the mean and median are close or noticeably different. A difference may suggest an uneven distribution, but the conclusion should be checked against the full statistical summary.

## 8.2 Transaction Amount

```python
amount = transactions["amount"]

amount_mean = amount.mean()
amount_median = amount.median()
amount_mode = amount.mode()

print(f"Mean Transaction Amount: {amount_mean:.2f}")
print(f"Median Transaction Amount: {amount_median:.2f}")
print("Mode Transaction Amount(s):", amount_mode.tolist()[:10])
```

Interpret the mean and median together. Do not assume that either measure is automatically the better representation without examining the distribution and possible unusual values.

---

# 9. Measures of Variability

## 9.1 Customer Age

```python
age_min = age.min()
age_max = age.max()
age_range = age_max - age_min
age_variance = age.var()
age_std = age.std()

print(f"Minimum Age: {age_min:.2f}")
print(f"Maximum Age: {age_max:.2f}")
print(f"Age Range: {age_range:.2f}")
print(f"Age Variance: {age_variance:.2f}")
print(f"Age Standard Deviation: {age_std:.2f}")
```

## 9.2 Transaction Amount

```python
amount_min = amount.min()
amount_max = amount.max()
amount_range = amount_max - amount_min
amount_variance = amount.var()
amount_std = amount.std()

print(f"Minimum Transaction Amount: {amount_min:.2f}")
print(f"Maximum Transaction Amount: {amount_max:.2f}")
print(f"Transaction Amount Range: {amount_range:.2f}")
print(f"Transaction Amount Variance: {amount_variance:.2f}")
print(f"Transaction Amount Standard Deviation: {amount_std:.2f}")
```

A larger standard deviation means values are more dispersed around the mean. It does not by itself prove instability, risk, or abnormal behavior.

---

# 10. Quartiles, IQR, and Potential Outliers

## 10.1 Five-Number Summary for Transaction Amount

```python
q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

print(f"Minimum: {amount.min():.2f}")
print(f"Q1: {q1:.2f}")
print(f"Median / Q2: {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"Maximum: {amount.max():.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Potential Outlier Lower Bound: {lower_bound:.2f}")
print(f"Potential Outlier Upper Bound: {upper_bound:.2f}")
```

## 10.2 Review Potential Transaction-Amount Outliers

```python
amount_outliers = transactions[
    (transactions["amount"] < lower_bound) |
    (transactions["amount"] > upper_bound)
].copy()

print("Potential transaction-amount outliers:", len(amount_outliers))
display(amount_outliers.head(20))
```

Values outside the IQR bounds should be **investigated**, not automatically deleted. A statistically unusual observation can still be a valid business event.

---

# 11. Frequency Distribution and Data Balance

## 11.1 Reusable Category Summary

```python
def category_summary(df, column):
    counts = df[column].value_counts(dropna=False)
    percentages = (
        df[column]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
    )

    return pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })
```

## 11.2 Review Major Categorical Variables

```python
categorical_checks = [
    (demographics, "gender"),
    (demographics, "income_level"),
    (transactions, "product_category"),
    (transactions, "payment_method"),
    (social, "platform"),
    (social, "interaction_type"),
    (social, "sentiment")
]

for df, column in categorical_checks:
    print(f"\n--- {column} ---")
    display(category_summary(df, column))
```

For each variable, identify:

- the most frequent category;
- the least represented category;
- whether the distribution appears relatively balanced or uneven; and
- whether the distribution may affect later analysis or machine learning.

A dominant category means it appears most frequently in the supplied data. It does not automatically prove customer preference or business performance.

---

# 12. Grouped Data Analysis

Grouped analysis helps reveal differences that may be hidden by overall averages.

## 12.1 Product Category and Transaction Amount

```python
product_amount_summary = (
    transactions
    .groupby("product_category")["amount"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max"
    )
    .round(2)
    .sort_values("mean", ascending=False)
)

display(product_amount_summary)
```

Questions to consider:

- Do typical transaction amounts differ across product categories?
- Are some categories more variable than others?
- Are group sizes large enough for a meaningful comparison?

## 12.2 Payment Method and Transaction Amount

```python
payment_summary = (
    transactions
    .groupby("payment_method")["amount"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std"
    )
    .round(2)
    .sort_values("mean", ascending=False)
)

display(payment_summary)
```

Do not interpret differences as causal. The table only shows how recorded transaction values differ across payment-method groups.

## 12.3 Platform and Sentiment

```python
platform_sentiment_count = pd.crosstab(
    social["platform"],
    social["sentiment"]
)

platform_sentiment_percent = (
    pd.crosstab(
        social["platform"],
        social["sentiment"],
        normalize="index"
    )
    .mul(100)
    .round(2)
)

print("Sentiment counts by platform:")
display(platform_sentiment_count)

print("\nSentiment percentages within each platform:")
display(platform_sentiment_percent)
```

This allows platforms with different interaction counts to be compared using both raw frequency and within-platform percentages.

---

# 13. Temporal Trend Analysis

Time-based analysis can identify changes or recurring patterns without requiring visualization.

## 13.1 Monthly Transaction Activity

```python
transactions["transaction_month"] = (
    transactions["transaction_date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_transactions = (
    transactions
    .groupby("transaction_month")
    .agg(
        transaction_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        median_amount=("amount", "median")
    )
    .reset_index()
)

display(monthly_transactions)
```

Review whether transaction count, total amount, or median amount changes across the available observation period.

## 13.2 Monthly Social Media Activity

```python
social["interaction_month"] = (
    social["interaction_date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_social = (
    social
    .groupby("interaction_month")
    .size()
    .reset_index(name="interaction_count")
)

display(monthly_social)
```

Describe only trends supported by the available period. Do not assume that transaction activity and social activity cause one another.

---

# 14. Customer-Level Cross-Dataset Analysis

Raw transactions and raw social interactions should not be directly joined to one another by `customer_id` because customers may have multiple records in both datasets.

Aggregate each event dataset to customer level first.

## 14.1 Customer-Level Transaction Measures

```python
customer_transactions = (
    transactions
    .groupby("customer_id")
    .agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        average_spend=("amount", "mean"),
        median_spend=("amount", "median")
    )
    .reset_index()
)

display(customer_transactions.head())
```

## 14.2 Customer-Level Social Measures

```python
customer_social = (
    social
    .groupby("customer_id")
    .agg(
        interaction_count=("interaction_id", "count")
    )
    .reset_index()
)

display(customer_social.head())
```

## 14.3 Build the Customer Analysis Table

```python
customer_analysis = (
    demographics
    .merge(customer_transactions, on="customer_id", how="left")
    .merge(customer_social, on="customer_id", how="left")
)

# Counts can be set to zero when the clean demographics customer has no
# corresponding recorded activity in the cleaned event tables.
customer_analysis["transaction_count"] = (
    customer_analysis["transaction_count"].fillna(0)
)

customer_analysis["interaction_count"] = (
    customer_analysis["interaction_count"].fillna(0)
)

display(customer_analysis.head())
print("Customer-level analytical shape:", customer_analysis.shape)
```

**Analytical grain:** one row represents one customer from the cleaned demographics dataset.

---

# 15. Cross-Dataset Grouped and Relationship Analysis

## 15.1 Income Level and Customer Spending

```python
income_spending = (
    customer_analysis
    .groupby("income_level")["total_spend"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std"
    )
    .round(2)
)

display(income_spending)
```

Compare mean, median, variability, and group size before describing any difference between income groups.

## 15.2 Age and Customer Spending

```python
age_spend_data = (
    customer_analysis[["age", "total_spend"]]
    .dropna()
)

age_spend_corr = age_spend_data.corr().iloc[0, 1]

print(f"Age vs Total Spend Pearson correlation: {age_spend_corr:.3f}")
```

Pearson correlation measures **linear association**. It does not establish that age causes a change in spending.

## 15.3 Social Interaction and Transaction Activity

```python
engagement_transaction_corr = (
    customer_analysis[
        ["interaction_count", "transaction_count"]
    ]
    .corr()
    .iloc[0, 1]
)

print(
    "Interaction Count vs Transaction Count Pearson correlation:",
    f"{engagement_transaction_corr:.3f}"
)
```

Interpret the direction and magnitude cautiously. Association does not prove that social activity causes transactions.

---

# 16. Numerical Correlation Review

```python
candidate_numeric = [
    "age",
    "transaction_count",
    "total_spend",
    "average_spend",
    "median_spend",
    "interaction_count"
]

numeric_columns = [
    column
    for column in candidate_numeric
    if column in customer_analysis.columns
]

correlation_matrix = customer_analysis[numeric_columns].corr()

display(correlation_matrix.round(3))
```

Do not include identifier columns such as `customer_id` in the correlation matrix.

Use correlation only where the variables and analytical question make sense.

---

# 17. Pattern, Trend, and Anomaly Interpretation

For each major result, use:

**Business Question → Analytical Method → Result → Interpretation**

For major findings, use:

**Finding → Evidence → Interpretation → Machine Learning Relevance**

Example:

```text
Finding:
[State the observed pattern.]

Evidence:
[Give the actual statistic, frequency, grouped result, trend, or outlier count.]

Interpretation:
[Explain what the result means within the supplied FinMark datasets.]

Machine Learning Relevance:
[Explain whether the pattern, imbalance, variability, or limitation may matter
for later feature selection, model preparation, validation, or evaluation.]

Limitation:
[State what the evidence does not establish.]
```

Only include conclusions supported by executed notebook results.

---

# 18. Machine Learning Readiness Assessment

Keep this section concise. Milestone 1 does not need to select or train a model.

Review:

## 18.1 Data Quality

- Are the cleaned datasets free of unexpected missing values and exact duplicates?
- Are identifier fields valid?
- Are dates usable?
- Do questionable values remain?

## 18.2 Data Balance

- Are categorical groups heavily uneven?
- Could underrepresented categories affect later analysis or model evaluation?

## 18.3 Outliers and Variability

- Do statistical outliers remain?
- Are they plausible observations or unresolved issues?
- Could wide variability require scaling, transformation, or robust methods later?

## 18.4 Feature Considerations

Potentially useful analytical variables may include:

- `age`
- `income_level`
- `product_category`
- `payment_method`
- `amount`
- `transaction_count`
- `total_spend`
- `average_spend`
- `interaction_count`
- `sentiment`
- derived temporal variables

Identifiers such as `customer_id`, `transaction_id`, and `interaction_id` should remain relationship or traceability keys unless a later modeling design provides a justified use.

---

# 19. Key Findings and Recommendations

Summarize only findings already established by the analysis.

## 19.1 Data Quality

- [Evidence-supported finding]
- [Evidence-supported finding]

## 19.2 Statistical Characteristics

- [Evidence-supported finding]
- [Evidence-supported finding]

## 19.3 Frequency and Balance

- [Evidence-supported finding]

## 19.4 Patterns and Relationships

- [Evidence-supported finding]

## 19.5 Anomalies or Remaining Issues

- [Evidence-supported issue or limitation]

## 19.6 Considerations for the Next Stage

- carry forward the strongest evidence-supported patterns for Milestone 2;
- investigate unresolved data limitations where necessary; and
- retain relevant statistical findings for later machine learning planning.

Do not define the final visualization design in Milestone 1.

---

# 20. Limitations

Discuss limitations supported by the actual analysis, such as:

- preprocessing reduced the available sample;
- independent cleaning created cross-dataset coverage gaps;
- some activity records cannot be linked to the cleaned demographics table;
- association does not establish causation;
- category imbalance may affect comparisons;
- customer-level aggregation changes the unit of analysis;
- available variables may not explain why observed behaviors occur; and
- findings apply only to the supplied FinMark datasets.

---

# 21. Conclusion

The conclusion should answer:

1. What did the EDA confirm about the structure and quality of the FinMark datasets?
2. What were the most important statistical characteristics?
3. What distributions, patterns, relationships, anomalies, or outliers were identified?
4. What issues may affect later machine learning activities?
5. What findings should be carried forward to Milestone 2?

Do not introduce new evidence in the conclusion.

---

# 22. Mapping to the Six AI-Graded Questions

| AI-Graded Question | Main Evidence |
|---|---|
| **Q1. Describe the datasets used in EDA** | Sections 1, 3, 4, 5, and Week 3 records |
| **Q2. Discuss preprocessing procedures** | Section 6 and Week 2–3 preprocessing documentation |
| **Q3. Discuss descriptive statistics** | Sections 7–10 |
| **Q4. Analyze frequency and value distributions** | Sections 10–13 |
| **Q5. Discuss patterns, relationships, trends, anomalies, or outliers** | Sections 12–17 |
| **Q6. Summarize findings and recommendations** | Sections 18–21 |

---

# 23. Final Analytical Chain

```text
FINMARK BUSINESS PROBLEM
        ↓
FINAL WEEK 3 PREPROCESSED DATA
        ↓
POST-CLEANING VALIDATION
        ↓
VARIABLE CLASSIFICATION
        ↓
CENTRAL TENDENCY
        ↓
VARIABILITY
        ↓
QUARTILES / IQR / OUTLIERS
        ↓
FREQUENCY AND BALANCE
        ↓
GROUPED ANALYSIS
        ↓
TEMPORAL ANALYSIS
        ↓
CUSTOMER-LEVEL INTEGRATION
        ↓
RELATIONSHIPS / PATTERNS / ANOMALIES
        ↓
FINDING
        ↓
EVIDENCE
        ↓
INTERPRETATION
        ↓
MACHINE LEARNING RELEVANCE
        ↓
LIMITATIONS
        ↓
MILESTONE 2 HANDOFF
```

---

# 24. Final Review Checklist

Before submission, verify that:

- [ ] The FinMark business problem is clearly defined.
- [ ] All three cleaned datasets are described.
- [ ] Numerical and categorical variables are identified.
- [ ] Week 2–3 preprocessing decisions are summarized.
- [ ] Missing-value and duplicate handling are explained.
- [ ] Data-type corrections are documented.
- [ ] Post-cleaning validation is confirmed.
- [ ] Mean, median, and mode are calculated where appropriate.
- [ ] Minimum, maximum, range, variance, and standard deviation are reviewed.
- [ ] Quartiles and IQR are reviewed.
- [ ] Potential outliers are investigated.
- [ ] Frequency distributions are analyzed.
- [ ] Data balance or imbalance is discussed.
- [ ] Grouped analysis is included.
- [ ] Temporal trends are reviewed where applicable.
- [ ] Cross-dataset integration preserves the correct analytical grain.
- [ ] Relationship analysis avoids causal claims.
- [ ] Every major finding is supported by notebook evidence.
- [ ] Findings are interpreted in the FinMark context.
- [ ] Machine learning considerations remain evidence-based.
- [ ] Limitations are documented.
- [ ] Questions 1–6 can be answered directly from notebook outputs.
- [ ] Milestone 2 visualization work is not duplicated in this milestone.
