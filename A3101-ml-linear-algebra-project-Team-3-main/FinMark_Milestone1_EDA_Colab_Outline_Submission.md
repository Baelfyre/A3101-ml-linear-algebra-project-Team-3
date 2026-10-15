# MO-IT162 Milestone 1: ML Solution EDA

**Project:** FinMark Corporation  
**Course:** MO-IT162 Math for Machine Learning: Linear Algebra  
**Milestone:** Milestone 1 - ML Solution EDA  
**Environment:** Google Colab / Jupyter Notebook  
**Libraries:** Pandas, NumPy, Matplotlib, Seaborn

## Project Overview

This notebook documents the Exploratory Data Analysis (EDA) of FinMark Corporation's finalized preprocessed datasets from Week 3. The analysis focuses on customer demographics, transaction activity, and social media interactions to assess data quality, summarize important statistical characteristics, identify patterns and anomalies, examine relevant relationships, and determine whether the available data is suitable for succeeding visualization and machine learning activities.

The notebook preserves a clear analytical flow from problem definition and data verification through statistical analysis, interpretation, and machine learning readiness assessment.

---

# 1. Business Problem and Analytical Needs

## 1.1 FinMark Business Problem
FinMark Corporation needs to improve how it analyzes customer information and generates useful insights from customer demographics, transaction activity, and social media interactions.

Before future machine learning activities can begin, the available datasets must first be examined to determine:

- whether the data is accurate, consistent, and usable;
- what statistical patterns and distributions are present;
- whether important differences exist across customer groups;
- whether anomalies or unusual observations remain;
- which relationships may be useful for later visualization and modeling.

## 1.2 Stakeholders and User Needs
Primary stakeholders include:

- FinMark decision-makers;
- FinMark data analysts; and
- the future machine learning development team.

Their analytical needs include:

- reliable customer and transaction data;
- understandable summaries of customer behavior;
- identification of possible data-quality risks;
- evidence-supported customer and market insights; and
- a validated analytical dataset that can support Milestone 2 and later machine learning work.

## 1.3 Objectives

The analysis aims to:

- verify the quality and usability of the finalized Week 3 datasets;
- summarize numerical and categorical variables using appropriate statistical methods;
- identify meaningful distributions, group differences, patterns, trends, and anomalies;
- examine relationships between relevant variables;
- interpret findings in the context of FinMark's analytical needs; and
- assess considerations that should be carried forward into Milestone 2 and future machine learning development.

## 1.4 Scope

### Included

- customer demographics;
- customer transactions;
- social media interactions;
- data verification and preprocessing review;
- descriptive statistics;
- frequency and distribution analysis;
- grouped and temporal analysis;
- cross-dataset customer-level analysis;
- exploratory visualizations;
- interpretation, limitations, and machine learning readiness.

### Outside the Scope

- machine learning model training;
- model evaluation or prediction;
- production deployment;
- causal claims that are not supported by the available data.


---

# 2. Data Collection and Dataset Overview

## 2.1 Data Sources
The analysis uses the three datasets supplied for the FinMark project:

1. **Customer Demographics**  
   Contains customer-level descriptive information.

2. **Customer Transactions**  
   Contains transaction-level purchasing activity.

3. **Social Media Interactions**  
   Contains interaction-level customer engagement information.

These datasets are directly relevant because together they represent the main customer characteristics, purchasing behavior, and engagement activity required by the FinMark problem.

## 2.2 Load the Finalized Week 3 Datasets

```python
# =============================================================================
# SECTION 2: LOAD FINALIZED WEEK 3 DATASETS
# =============================================================================
# Replace the placeholder filenames below with the exact filenames submitted
# during Week 3. The final EDA should use the finalized preprocessed datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

demographics = pd.read_csv("/content/REPLACE_WITH_FINAL_DEMOGRAPHICS_FILE.csv")
transactions = pd.read_csv("/content/REPLACE_WITH_FINAL_TRANSACTIONS_FILE.csv")
social = pd.read_csv("/content/REPLACE_WITH_FINAL_SOCIAL_MEDIA_FILE.csv")

# Confirm that the three datasets loaded successfully.
print("Demographics:", demographics.shape)
print("Transactions:", transactions.shape)
print("Social Media:", social.shape)
```

## 2.3 Dataset Structure and Relevance

```python
# =============================================================================
# SECTION 2.3: REVIEW DATASET STRUCTURE
# =============================================================================
# Review shape, columns, data types, and unique values to document the structure
# of each dataset and confirm that the required variables are available.

datasets = {
    "Customer Demographics": demographics,
    "Customer Transactions": transactions,
    "Social Media Interactions": social
}

for name, df in datasets.items():
    print(f"\n{name}")
    print("-" * len(name))
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nData types:")
    display(df.dtypes)
    print("\nUnique values per column:")
    display(df.nunique())
```

## 2.4 Variable Classification

Expected classifications include:

| Dataset | Variable | Variable Type | Analytical Role |
|---|---|---|---|
| Demographics | CustomerID | Identifier | Relationship key |
| Demographics | Age | Quantitative | Ratio |
| Demographics | Gender | Categorical | Nominal |
| Demographics | IncomeLevel | Categorical | Ordinal if ordered categories are validated |
| Demographics | SignupDate | Temporal | Date |
| Transactions | TransactionID | Identifier | Record key |
| Transactions | CustomerID | Identifier | Relationship key |
| Transactions | TransactionDate | Temporal | Date |
| Transactions | Amount | Quantitative | Ratio |
| Transactions | ProductCategory | Categorical | Nominal |
| Transactions | PaymentMethod | Categorical | Nominal |
| Social Media | InteractionID | Identifier | Record key |
| Social Media | CustomerID | Identifier | Relationship key |
| Social Media | InteractionDate | Temporal | Date |
| Social Media | Platform | Categorical | Nominal |
| Social Media | InteractionType | Categorical | Nominal |
| Social Media | Sentiment | Categorical | Ordinal only if a valid ordered scale is confirmed |

---

# 3. Data Verification

The purpose of this section is to demonstrate that the team systematically checked the accuracy, structure, reliability, and usability of the finalized data.

## 3.1 Missing Values

```python
# =============================================================================
# SECTION 3.1: VERIFY REMAINING MISSING VALUES
# =============================================================================
# Missing values should be reviewed after preprocessing to confirm whether
# important analytical fields remain incomplete.

for name, df in datasets.items():
    print(f"\n{name} - Missing Values")
    display(df.isna().sum())
```

## 3.2 Exact Duplicate Rows

```python
# =============================================================================
# SECTION 3.2: VERIFY EXACT DUPLICATE ROWS
# =============================================================================
# Exact duplicate rows should normally have been handled during preprocessing.
# Any remaining duplicate should be investigated before further analysis.

for name, df in datasets.items():
    print(name, "exact duplicates:", df.duplicated().sum())
```

## 3.3 Identifier Integrity

```python
# =============================================================================
# SECTION 3.3: VERIFY UNIQUE IDENTIFIERS
# =============================================================================
# Repeated IDs are checked separately from exact duplicates because a repeated
# identifier may contain conflicting information.

print(
    "Duplicated demographic CustomerIDs:",
    demographics["CustomerID"].duplicated().sum()
)

print(
    "Duplicated TransactionIDs:",
    transactions["TransactionID"].duplicated().sum()
)

print(
    "Duplicated InteractionIDs:",
    social["InteractionID"].duplicated().sum()
)
```

## 3.4 Data Type Verification

```python
# =============================================================================
# SECTION 3.4: VERIFY DATA TYPES
# =============================================================================
# Statistical variable type and Pandas storage type are not always the same.
# Verify that quantitative values are numeric and date fields are usable.

print("Demographics dtypes:")
display(demographics.dtypes)

print("\nTransactions dtypes:")
display(transactions.dtypes)

print("\nSocial Media dtypes:")
display(social.dtypes)
```

## 3.5 Date Verification

```python
# =============================================================================
# SECTION 3.5: VERIFY DATE VARIABLES
# =============================================================================
# Convert date variables to datetime and count any values that cannot be parsed.

date_columns = [
    (demographics, "SignupDate"),
    (transactions, "TransactionDate"),
    (social, "InteractionDate")
]

for df, column in date_columns:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")
        print(column, "missing/unparseable dates:", df[column].isna().sum())
```

## 3.6 Numerical Range and Logical Checks

```python
# =============================================================================
# SECTION 3.6: VERIFY NUMERICAL RANGES
# =============================================================================
# Review the minimum and maximum values of major numerical fields.
# These checks help determine whether impossible or questionable values remain.

if "Age" in demographics.columns:
    print("Age minimum:", demographics["Age"].min())
    print("Age maximum:", demographics["Age"].max())

if "Amount" in transactions.columns:
    print("Amount minimum:", transactions["Amount"].min())
    print("Amount maximum:", transactions["Amount"].max())
```

## 3.7 Category Verification

```python
# =============================================================================
# SECTION 3.7: VERIFY CATEGORICAL VALUES
# =============================================================================
# Review category values for inconsistent spelling, unexpected labels,
# missing categories, or unresolved formatting differences.

categorical_checks = [
    (demographics, "Gender"),
    (demographics, "IncomeLevel"),
    (transactions, "ProductCategory"),
    (transactions, "PaymentMethod"),
    (social, "Platform"),
    (social, "InteractionType"),
    (social, "Sentiment")
]

for df, column in categorical_checks:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))
```

## 3.8 Cross-Dataset Relationship Integrity

```python
# =============================================================================
# SECTION 3.8: VERIFY CUSTOMER IDS ACROSS DATASETS
# =============================================================================
# Confirm that transaction and social-media customer IDs correspond to
# customers represented in the demographics dataset.

demographic_ids = set(demographics["CustomerID"].dropna())
transaction_ids = set(transactions["CustomerID"].dropna())
social_ids = set(social["CustomerID"].dropna())

print(
    "Transaction CustomerIDs not found in demographics:",
    len(transaction_ids - demographic_ids)
)

print(
    "Social CustomerIDs not found in demographics:",
    len(social_ids - demographic_ids)
)
```

## 3.9 Verification Summary
Create a concise table containing:

- issue checked;
- dataset affected;
- verification result;
- whether the issue was already resolved;
- whether further treatment is still required.

---

# 4. Data Cleaning and Preprocessing Review

This section documents the actual preprocessing completed during Weeks 2–3.

For every important cleaning decision, use:

**Issue → Treatment → Reason → Validation**

## 4.1 Duplicate Handling
Explain:

- how exact duplicates were identified;
- whether exact duplicate rows were removed;
- how repeated IDs with conflicting information were treated;
- why the treatment was appropriate.

## 4.2 Missing Values
Explain:

- which variables contained missing values;
- how missing values were identified;
- whether values were retained, removed, or imputed;
- why that treatment was chosen.

Do not claim that all missing values must be removed or filled.

## 4.3 Data Type Corrections
Explain corrections such as:

- numeric conversion for Age or Amount;
- date conversion for temporal variables;
- category normalization where required.

## 4.4 Invalid or Questionable Values
Explain the treatment of values identified during preprocessing, such as:

- invalid ages;
- non-numeric age labels;
- unusual transaction amount entries;
- zero or negative transaction values;
- unexpected categorical values.

Do not silently convert or delete questionable values without justification.

## 4.5 Category Standardization
If applicable, document:

- capitalization normalization;
- spelling correction;
- whitespace removal;
- preservation of legitimate categories.

## 4.6 Post-Cleaning Validation

```python
# =============================================================================
# SECTION 4.6: POST-CLEANING VALIDATION
# =============================================================================
# Re-run the major quality checks after preprocessing.
# This proves that the intended cleaning actions produced the expected result.

for name, df in datasets.items():
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Exact duplicates:", df.duplicated().sum())
    print("Missing values:")
    display(df.isna().sum())
    print("Data types:")
    display(df.dtypes)
```

## 4.7 Cleaning Decision Table

| Issue | Treatment | Reason | Validation |
|---|---|---|---|
| [Actual issue] | [Actual treatment] | [Why] | [Post-cleaning evidence] |

---

# 5. Exploratory Data Analysis

The analysis should follow:

**Business Question → Analytical Method → Result → Interpretation**

Statistical methods should be chosen because they address a specific analytical question.

## 5.1 Central Tendency

```python
# =============================================================================
# SECTION 5.1: CENTRAL TENDENCY
# =============================================================================
# Mean describes the overall average.
# Median describes the middle value.
# Mode identifies the most frequently occurring value.
# Apply these only where they are statistically meaningful.

age = demographics["Age"].dropna()
amount = transactions["Amount"].dropna()

print("Age")
print("Mean:", round(age.mean(), 2))
print("Median:", round(age.median(), 2))
print("Mode:", age.mode().tolist())

print("\nTransaction Amount")
print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Mode:", amount.mode().tolist()[:10])
```

## 5.2 Variability

```python
# =============================================================================
# SECTION 5.2: VARIABILITY OF TRANSACTION AMOUNT
# =============================================================================
# Minimum and maximum show the observed extremes.
# Range measures the distance between them.
# Variance and standard deviation measure spread around the mean.

amount_min = amount.min()
amount_max = amount.max()
amount_range = amount_max - amount_min
amount_variance = amount.var()
amount_std = amount.std()

print("Minimum:", round(amount_min, 2))
print("Maximum:", round(amount_max, 2))
print("Range:", round(amount_range, 2))
print("Variance:", round(amount_variance, 2))
print("Standard Deviation:", round(amount_std, 2))
```

## 5.3 Quartiles, IQR, and Potential Outliers

```python
# =============================================================================
# SECTION 5.3: QUARTILES AND IQR
# =============================================================================
# Quartiles describe the relative position of transaction values.
# IQR measures the spread of the middle 50% and helps flag possible outliers.

q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

print("Q1:", round(q1, 2))
print("Median / Q2:", round(q2, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower Bound:", round(lower_bound, 2))
print("Upper Bound:", round(upper_bound, 2))

amount_outliers = transactions[
    (transactions["Amount"] < lower_bound) |
    (transactions["Amount"] > upper_bound)
]

print("Potential outliers:", len(amount_outliers))
display(amount_outliers.head(20))
```

## 5.4 Numerical Distribution Visualizations

```python
# =============================================================================
# SECTION 5.4A: AGE DISTRIBUTION
# =============================================================================
# The histogram shows where customer ages are concentrated.
# Compare the distribution with the calculated mean and median.

plt.figure(figsize=(9, 5))
sns.histplot(data=demographics, x="Age", bins=15, kde=True)
plt.axvline(age.mean(), linestyle="--", label=f"Mean = {age.mean():.2f}")
plt.axvline(age.median(), linestyle=":", label=f"Median = {age.median():.2f}")
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

```python
# =============================================================================
# SECTION 5.4B: TRANSACTION AMOUNT DISTRIBUTION
# =============================================================================
# This histogram shows the concentration and spread of transaction amounts.

plt.figure(figsize=(9, 5))
sns.histplot(data=transactions, x="Amount", bins=20, kde=True)
plt.axvline(amount.mean(), linestyle="--", label=f"Mean = {amount.mean():.2f}")
plt.axvline(amount.median(), linestyle=":", label=f"Median = {amount.median():.2f}")
plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

```python
# =============================================================================
# SECTION 5.4C: TRANSACTION AMOUNT BOX PLOT
# =============================================================================
# The box plot summarizes median, quartiles, spread, and potential outliers.

plt.figure(figsize=(9, 4))
sns.boxplot(data=transactions, x="Amount")
plt.title("Box Plot of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.show()
```

## 5.5 Categorical Frequency Distributions

```python
# =============================================================================
# SECTION 5.5: CATEGORICAL FREQUENCY DISTRIBUTIONS
# =============================================================================
# Count and percentage are reviewed together to identify dominant,
# underrepresented, or potentially imbalanced categories.

def category_summary(df, column):
    counts = df[column].value_counts(dropna=False)
    percentages = (
        df[column]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
    )

    return pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

for df, column in categorical_checks:
    if column in df.columns:
        print(f"\n{column}")
        display(category_summary(df, column))
```

```python
# =============================================================================
# SECTION 5.5A: VISUALIZE PRODUCT CATEGORY FREQUENCY
# =============================================================================
# Frequency shows how often each category appears in recorded transactions.

order = transactions["ProductCategory"].value_counts().index

plt.figure(figsize=(9, 5))
sns.countplot(
    data=transactions,
    y="ProductCategory",
    order=order
)
plt.title("Transaction Frequency by Product Category")
plt.xlabel("Number of Transactions")
plt.ylabel("Product Category")
plt.show()
```

```python
# =============================================================================
# SECTION 5.5B: VISUALIZE SOCIAL PLATFORM FREQUENCY
# =============================================================================
# Higher frequency means more recorded interactions for the platform.
# It does not by itself prove preference or effectiveness.

platform_order = social["Platform"].value_counts().index

plt.figure(figsize=(9, 5))
sns.countplot(
    data=social,
    y="Platform",
    order=platform_order
)
plt.title("Social Media Interactions by Platform")
plt.xlabel("Number of Interactions")
plt.ylabel("Platform")
plt.show()
```

## 5.6 Grouped Analysis

```python
# =============================================================================
# SECTION 5.6A: TRANSACTION AMOUNT BY PRODUCT CATEGORY
# =============================================================================
# Grouped analysis compares typical transaction values and variability across
# product categories instead of relying only on the overall average.

product_summary = (
    transactions
    .groupby("ProductCategory")["Amount"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Standard_Deviation="std",
        Minimum="min",
        Maximum="max"
    )
    .round(2)
    .sort_values("Mean", ascending=False)
)

display(product_summary)
```

```python
# =============================================================================
# SECTION 5.6B: VISUALIZE MEDIAN TRANSACTION AMOUNT BY PRODUCT CATEGORY
# =============================================================================

category_median = (
    transactions
    .groupby("ProductCategory", as_index=False)["Amount"]
    .median()
    .sort_values("Amount", ascending=False)
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=category_median,
    x="Amount",
    y="ProductCategory"
)
plt.title("Median Transaction Amount by Product Category")
plt.xlabel("Median Transaction Amount")
plt.ylabel("Product Category")
plt.show()
```

```python
# =============================================================================
# SECTION 5.6C: TRANSACTION AMOUNT BY PAYMENT METHOD
# =============================================================================

payment_summary = (
    transactions
    .groupby("PaymentMethod")["Amount"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Standard_Deviation="std"
    )
    .round(2)
)

display(payment_summary)
```

```python
# =============================================================================
# SECTION 5.6D: SENTIMENT DISTRIBUTION BY PLATFORM
# =============================================================================
# Row percentages make platforms with different interaction counts comparable.

platform_sentiment = pd.crosstab(
    social["Platform"],
    social["Sentiment"],
    normalize="index"
).mul(100).round(2)

display(platform_sentiment)

platform_sentiment.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title("Sentiment Distribution by Social Media Platform")
plt.xlabel("Platform")
plt.ylabel("Percentage of Interactions")
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

## 5.7 Temporal Trend Analysis

```python
# =============================================================================
# SECTION 5.7A: MONTHLY TRANSACTION ACTIVITY
# =============================================================================
# Monthly aggregation helps identify changes or recurring trends over time.

transactions["TransactionMonth"] = (
    transactions["TransactionDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_transactions = (
    transactions
    .groupby("TransactionMonth")
    .agg(
        TransactionCount=("TransactionID", "count"),
        TotalAmount=("Amount", "sum"),
        MedianAmount=("Amount", "median")
    )
    .reset_index()
)

display(monthly_transactions)

plt.figure(figsize=(11, 5))
sns.lineplot(
    data=monthly_transactions,
    x="TransactionMonth",
    y="TransactionCount",
    marker="o"
)
plt.title("Monthly Transaction Activity")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# =============================================================================
# SECTION 5.7B: MONTHLY SOCIAL MEDIA ACTIVITY
# =============================================================================

social["InteractionMonth"] = (
    social["InteractionDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_social = (
    social
    .groupby("InteractionMonth")
    .size()
    .reset_index(name="InteractionCount")
)

display(monthly_social)

plt.figure(figsize=(11, 5))
sns.lineplot(
    data=monthly_social,
    x="InteractionMonth",
    y="InteractionCount",
    marker="o"
)
plt.title("Monthly Social Media Interaction Activity")
plt.xlabel("Month")
plt.ylabel("Number of Interactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 5.8 Cross-Dataset Customer-Level Analysis

Raw transaction and social interaction records should not be directly merged event-to-event because one customer can have multiple records in both datasets.

```python
# =============================================================================
# SECTION 5.8A: AGGREGATE TRANSACTIONS BY CUSTOMER
# =============================================================================

customer_transactions = (
    transactions
    .groupby("CustomerID")
    .agg(
        TransactionCount=("TransactionID", "count"),
        TotalSpend=("Amount", "sum"),
        AverageSpend=("Amount", "mean"),
        MedianSpend=("Amount", "median")
    )
    .reset_index()
)

display(customer_transactions.head())
```

```python
# =============================================================================
# SECTION 5.8B: AGGREGATE SOCIAL INTERACTIONS BY CUSTOMER
# =============================================================================

customer_social = (
    social
    .groupby("CustomerID")
    .agg(
        InteractionCount=("InteractionID", "count")
    )
    .reset_index()
)

display(customer_social.head())
```

```python
# =============================================================================
# SECTION 5.8C: BUILD CUSTOMER-LEVEL ANALYTICAL DATASET
# =============================================================================

customer_analysis = (
    demographics
    .merge(customer_transactions, on="CustomerID", how="left")
    .merge(customer_social, on="CustomerID", how="left")
)

customer_analysis["TransactionCount"] = (
    customer_analysis["TransactionCount"].fillna(0)
)

customer_analysis["InteractionCount"] = (
    customer_analysis["InteractionCount"].fillna(0)
)

print("Customer-level analysis shape:", customer_analysis.shape)
display(customer_analysis.head())
```

## 5.9 Relationship Analysis

```python
# =============================================================================
# SECTION 5.9A: INCOME LEVEL VS CUSTOMER TOTAL SPENDING
# =============================================================================

income_spending = (
    customer_analysis
    .groupby("IncomeLevel")["TotalSpend"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Standard_Deviation="std"
    )
    .round(2)
)

display(income_spending)

plt.figure(figsize=(9, 5))
sns.boxplot(
    data=customer_analysis,
    x="IncomeLevel",
    y="TotalSpend"
)
plt.title("Customer Total Spending by Income Level")
plt.xlabel("Income Level")
plt.ylabel("Total Spending")
plt.show()
```

```python
# =============================================================================
# SECTION 5.9B: AGE VS CUSTOMER TOTAL SPENDING
# =============================================================================
# Association does not establish that age causes differences in spending.

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=customer_analysis,
    x="Age",
    y="TotalSpend",
    alpha=0.6
)
plt.title("Customer Age vs Total Spending")
plt.xlabel("Age")
plt.ylabel("Total Spending")
plt.show()

age_spend_corr = (
    customer_analysis[["Age", "TotalSpend"]]
    .corr()
    .iloc[0, 1]
)

print("Age vs TotalSpend correlation:", round(age_spend_corr, 3))
```

```python
# =============================================================================
# SECTION 5.9C: SOCIAL ENGAGEMENT VS TRANSACTION ACTIVITY
# =============================================================================
# This explores whether customers with more recorded social interactions also
# tend to have more recorded transactions.

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=customer_analysis,
    x="InteractionCount",
    y="TransactionCount",
    alpha=0.6
)
plt.title("Social Interaction Count vs Transaction Count")
plt.xlabel("Social Interaction Count")
plt.ylabel("Transaction Count")
plt.show()

engagement_transaction_corr = (
    customer_analysis[["InteractionCount", "TransactionCount"]]
    .corr()
    .iloc[0, 1]
)

print(
    "InteractionCount vs TransactionCount correlation:",
    round(engagement_transaction_corr, 3)
)
```

```python
# =============================================================================
# SECTION 5.9D: CORRELATION MATRIX
# =============================================================================
# Only meaningful numerical variables are included.
# IDs are excluded because they are identifiers, not measured quantities.

candidate_numeric = [
    "Age",
    "TransactionCount",
    "TotalSpend",
    "AverageSpend",
    "MedianSpend",
    "InteractionCount"
]

numeric_columns = [
    col for col in candidate_numeric
    if col in customer_analysis.columns
]

correlation_matrix = customer_analysis[numeric_columns].corr()

display(correlation_matrix.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Matrix of Customer-Level Numerical Variables")
plt.tight_layout()
plt.show()
```

---

# 6. Data Assessment and Interpretation

This section converts analytical outputs into accurate conclusions and business implications.

Use this structure for every major finding:

**Finding → Evidence → Interpretation → Business Implication**

## 6.1 Most Important Findings

For each important result, document:

### Finding
What was observed?

### Evidence
Which statistic, table, grouped result, or visualization supports it?

### Interpretation
What does the evidence mean within the FinMark dataset?

### Business Implication
Why might the result matter to FinMark?

### Limitation
What does the evidence not prove?

## 6.2 Patterns and Trends
Discuss only patterns supported by the EDA, such as:

- differences between product categories;
- differences between income groups;
- transaction changes over time;
- social-media activity trends;
- differences in sentiment composition;
- customer-level transaction and engagement patterns.

## 6.3 Anomalies and Outliers
Distinguish between:

1. **Data-quality anomalies**
2. **Questionable business values**
3. **Statistical outliers**

Do not automatically remove statistical outliers.

## 6.4 Machine Learning Readiness
Assess:

- remaining missing data;
- remaining data imbalance;
- useful numerical and categorical features;
- identifiers that should not be treated as normal model features;
- possible scaling or transformation needs;
- potential issues caused by outliers;
- unresolved data-quality concerns;
- whether more data is needed before modeling.

## 6.5 Further Investigation
For the strongest findings, identify what additional evidence would help explain them.

Examples:

- What explains differences between customer segments?
- Are unusual transaction values legitimate refunds or promotions?
- Do observed patterns persist over a longer period?
- Are additional customer preference or marketing variables needed?
- Are category imbalances natural or caused by incomplete data collection?

## 6.6 Recommendations
Recommendations should follow directly from the EDA.

Possible recommendation types include:

- prioritize a pattern for Milestone 2 visualization;
- investigate a remaining data-quality issue;
- retain a derived customer-level feature for later modeling;
- obtain additional data needed to test a hypothesis;
- monitor a specific metric in future analysis.

Avoid unsupported operational prescriptions.

## 6.7 Success Measures
Where a recommendation can later be evaluated, define an appropriate metric.

Possible measures include:

- transaction frequency;
- total customer spending;
- average or median transaction amount;
- customer engagement frequency;
- sentiment distribution;
- changes by customer segment;
- later model evaluation metrics.

Do not invent arbitrary improvement targets unless a business baseline or requirement is provided.

---

# 7. Final Conclusion

The conclusion should summarize:

- the quality and usability of the finalized datasets;
- the most important statistical characteristics;
- the strongest patterns or relationships found;
- major anomalies or limitations;
- whether the data is ready for Milestone 2;
- what should be investigated further before future model development.

Do not introduce new findings in the conclusion.

---

# 8. Final Review Checklist

## Problem / Needs Identification
- [ ] Target problem is clearly and concisely defined.
- [ ] Scope is complete but not overly broad.
- [ ] Stakeholder/user needs are identified.
- [ ] Requirements are explicit.
- [ ] Constraints and limitations are stated.

## Data Collection
- [ ] All three relevant FinMark datasets are used.
- [ ] Dataset source is documented.
- [ ] Each dataset's purpose is explained.
- [ ] Relevant fields are identified.
- [ ] Dataset relationships are explained.
- [ ] No unsupported external data is introduced.

## Data Verification
- [ ] Dataset dimensions are checked.
- [ ] Column names and structure are reviewed.
- [ ] Data types are verified.
- [ ] Missing values are checked.
- [ ] Exact duplicates are checked.
- [ ] Identifier duplicates are checked.
- [ ] Invalid or questionable ranges are checked.
- [ ] Categories are reviewed.
- [ ] Dates are validated.
- [ ] Cross-dataset relationship integrity is checked.

## Data Cleaning
- [ ] Cleaning decisions from Weeks 2–3 are documented.
- [ ] Duplicate handling is justified.
- [ ] Missing-value treatment is justified.
- [ ] Data-type corrections are explained.
- [ ] Invalid/questionable values are documented.
- [ ] Category standardization is documented where applicable.
- [ ] Post-cleaning validation is shown.
- [ ] Raw and cleaned data states are not confused.

## Data Analysis
- [ ] Mean is calculated where appropriate.
- [ ] Median is calculated where appropriate.
- [ ] Mode is calculated where appropriate.
- [ ] Minimum and maximum are reviewed.
- [ ] Range is calculated.
- [ ] Variance is calculated.
- [ ] Standard deviation is calculated.
- [ ] Quartiles and IQR are analyzed.
- [ ] Numerical distributions are visualized.
- [ ] Categorical frequency distributions are analyzed.
- [ ] Grouped comparisons are performed.
- [ ] Temporal trends are analyzed where applicable.
- [ ] Cross-dataset customer-level analysis is included.
- [ ] Relationships are examined using appropriate methods.
- [ ] Analytical methods are tied to business questions.

## Data Assessment & Interpretation
- [ ] Every major finding is supported by evidence.
- [ ] Findings are interpreted accurately.
- [ ] Business implications are explained.
- [ ] Correlation is not treated as causation.
- [ ] Anomalies and outliers are interpreted carefully.
- [ ] Limitations are documented.
- [ ] ML readiness is assessed.
- [ ] Recommendations are evidence-based.
- [ ] Further investigation is identified where evidence is incomplete.

---

# 9. Final Analytical Flow

```text
FINMARK BUSINESS PROBLEM
        ↓
STAKEHOLDER NEEDS / REQUIREMENTS / CONSTRAINTS
        ↓
DATA COLLECTION
        ↓
DATA VERIFICATION
        ↓
DATA CLEANING REVIEW
        ↓
POST-CLEANING VALIDATION
        ↓
DESCRIPTIVE STATISTICS
        ↓
DISTRIBUTIONS / FREQUENCIES
        ↓
GROUPED ANALYSIS
        ↓
TEMPORAL ANALYSIS
        ↓
CROSS-DATASET CUSTOMER ANALYSIS
        ↓
RELATIONSHIPS / PATTERNS / ANOMALIES
        ↓
FINDING
        ↓
EVIDENCE
        ↓
INTERPRETATION
        ↓
BUSINESS IMPLICATION
        ↓
LIMITATION
        ↓
FURTHER INVESTIGATION
        ↓
RECOMMENDATION
        ↓
ML READINESS
        ↓
MILESTONE 2 DATAVIZ
```
