# MO-IT162 Milestone 1: ML Solution EDA
## Final Colab Outline and Analysis Guide

> **Purpose:** This document is the canonical outline for the FinMark Corporation Milestone 1 Exploratory Data Analysis (EDA).  
> It is designed for Google Colab / Jupyter Notebook use and follows the required workflow from the Week 3 preprocessed datasets through statistical analysis, interpretation, visualization, and machine learning readiness assessment.
>
> **Notebook convention:** Explanations are written as comments inside the same code block as the commands they explain.  
> Replace any placeholder cleaned-dataset filename with the exact filename submitted by the team during Week 3.

---

# 1. Project Context and Problem Identification

## 1.1 FinMark Business Problem

FinMark Corporation needs to improve how it analyzes customer information and generates useful insights from its available data. The company has collected information from three main sources: customer demographics, customer transactions, and social media interactions. Before these datasets can support future machine learning activities, their structure, quality, statistical properties, patterns, and relationships must first be examined.

Milestone 1 therefore focuses on Exploratory Data Analysis (EDA) to determine whether the preprocessed FinMark datasets are suitable for deeper analysis and later machine learning development.

## 1.2 Stakeholders and Analytical Needs

Primary stakeholders include:

- FinMark decision-makers
- Data analysts
- Future machine learning developers

The analysis should help them understand:

- whether the available data is sufficiently complete and consistent;
- what typical customer and transaction characteristics look like;
- how much numerical values vary;
- which categories dominate or are underrepresented;
- whether meaningful differences exist between customer groups;
- whether unusual values or anomalies remain;
- which relationships deserve further investigation; and
- which dataset characteristics may affect later machine learning activities.

## 1.3 Milestone 1 Objectives

The EDA will:

1. review the structure of the finalized Week 3 preprocessed datasets;
2. document the preprocessing decisions performed during Weeks 2–3;
3. classify numerical, categorical, identifier, and temporal variables;
4. calculate measures of central tendency;
5. calculate measures of variability;
6. examine quartiles, distributions, and potential outliers;
7. analyze categorical frequencies and possible imbalance;
8. compare meaningful groups using grouped analysis;
9. investigate relationships, patterns, and trends;
10. assess possible issues that may affect future machine learning activities; and
11. summarize the most important evidence-supported findings and recommendations.

## 1.4 Scope and Limitations

This milestone focuses on exploratory analysis and machine learning preparation.

It does **not** train or evaluate a machine learning model.

Observed relationships should be interpreted as associations unless stronger evidence supports a causal conclusion.

---

# 2. Import Libraries and Configure the Notebook

```python
# =============================================================================
# SECTION 2: IMPORT LIBRARIES
# =============================================================================
# Pandas is used for loading, cleaning, grouping, and summarizing tabular data.
# NumPy supports numerical calculations.
# Matplotlib and Seaborn are used for exploratory visualizations.
# The visualizations in Milestone 1 support interpretation rather than serve
# as the final polished dashboard, which belongs to Milestone 2.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display more columns when reviewing DataFrames in Colab.
pd.set_option("display.max_columns", None)

# Use a readable default chart size for exploratory figures.
plt.rcParams["figure.figsize"] = (9, 5)
```

---

# 3. Load the Finalized Week 3 Preprocessed Datasets

```python
# =============================================================================
# SECTION 3: LOAD FINALIZED PREPROCESSED DATASETS
# =============================================================================
# IMPORTANT:
# Replace these filenames with the exact finalized files submitted in Week 3.
# Do not use the original contaminated files for the final EDA unless a section
# explicitly compares raw and cleaned data for documentation purposes.

demographics_path = "/content/REPLACE_WITH_FINAL_DEMOGRAPHICS_FILE.csv"
transactions_path = "/content/REPLACE_WITH_FINAL_TRANSACTIONS_FILE.csv"
social_path = "/content/REPLACE_WITH_FINAL_SOCIAL_MEDIA_FILE.csv"

demographics = pd.read_csv(demographics_path)
transactions = pd.read_csv(transactions_path)
social = pd.read_csv(social_path)

# Confirm that all three datasets loaded successfully.
print("Demographics shape:", demographics.shape)
print("Transactions shape:", transactions.shape)
print("Social media shape:", social.shape)
```

---

# 4. Dataset Overview and Structure

## 4.1 Customer Demographics

```python
# =============================================================================
# SECTION 4.1: CUSTOMER DEMOGRAPHICS OVERVIEW
# =============================================================================
# Review the first rows to understand how the cleaned dataset is structured.
display(demographics.head())

# Check the number of rows and columns.
print("Shape:", demographics.shape)

# Review column names.
print("Columns:")
print(demographics.columns.tolist())

# Review data types and non-null counts.
demographics.info()

# Count unique values to help distinguish identifiers, categories,
# and variables with repeated observations.
print("\nUnique values per column:")
display(demographics.nunique())
```

## 4.2 Customer Transactions

```python
# =============================================================================
# SECTION 4.2: CUSTOMER TRANSACTIONS OVERVIEW
# =============================================================================
display(transactions.head())

print("Shape:", transactions.shape)

print("Columns:")
print(transactions.columns.tolist())

transactions.info()

print("\nUnique values per column:")
display(transactions.nunique())
```

## 4.3 Social Media Interactions

```python
# =============================================================================
# SECTION 4.3: SOCIAL MEDIA INTERACTIONS OVERVIEW
# =============================================================================
display(social.head())

print("Shape:", social.shape)

print("Columns:")
print(social.columns.tolist())

social.info()

print("\nUnique values per column:")
display(social.nunique())
```

---

# 5. Variable Classification

Use the actual cleaned columns to classify variables before selecting statistical methods.

Expected FinMark classifications include:

| Dataset | Variable | Statistical Type | Measurement / Analytical Role |
|---|---|---|---|
| Demographics | CustomerID | Identifier / qualitative | Nominal relationship key |
| Demographics | Age | Quantitative | Ratio |
| Demographics | Gender | Qualitative | Nominal |
| Demographics | IncomeLevel | Qualitative | Ordinal if ordered categories are validated |
| Demographics | SignupDate | Temporal | Date / interval-like |
| Transactions | TransactionID | Identifier / qualitative | Nominal record key |
| Transactions | CustomerID | Identifier / qualitative | Nominal relationship key |
| Transactions | TransactionDate | Temporal | Date / interval-like |
| Transactions | Amount | Quantitative | Ratio |
| Transactions | ProductCategory | Qualitative | Nominal |
| Transactions | PaymentMethod | Qualitative | Nominal |
| Social Media | InteractionID | Identifier / qualitative | Nominal record key |
| Social Media | CustomerID | Identifier / qualitative | Nominal relationship key |
| Social Media | InteractionDate | Temporal | Date / interval-like |
| Social Media | Platform | Qualitative | Nominal |
| Social Media | InteractionType | Qualitative | Nominal |
| Social Media | Sentiment | Qualitative | Ordinal only if an ordered sentiment scale is validated |

```python
# =============================================================================
# SECTION 5: VERIFY VARIABLE TYPES AFTER PREPROCESSING
# =============================================================================
# Pandas data type and statistical variable type are not always the same.
# For example, CustomerID may be numeric in storage but it is an identifier,
# so calculating its mean would not have analytical meaning.

print("Demographics dtypes:")
display(demographics.dtypes)

print("\nTransactions dtypes:")
display(transactions.dtypes)

print("\nSocial media dtypes:")
display(social.dtypes)
```

---

# 6. Preprocessing Review and Post-Cleaning Validation

This section summarizes the Weeks 2–3 preprocessing work and verifies that the final datasets remain analytically usable.

The reporting pattern should be:

**Issue → Treatment → Reason → Validation**

## 6.1 Missing Values

```python
# =============================================================================
# SECTION 6.1: CHECK REMAINING MISSING VALUES
# =============================================================================
# Missing values should be reviewed after preprocessing because some may have
# been intentionally retained rather than imputed without evidence.

print("Demographics missing values:")
display(demographics.isna().sum())

print("\nTransactions missing values:")
display(transactions.isna().sum())

print("\nSocial media missing values:")
display(social.isna().sum())
```

## 6.2 Duplicate Records

```python
# =============================================================================
# SECTION 6.2: CHECK DUPLICATE ROWS
# =============================================================================
# Exact duplicates should have been addressed during preprocessing.
# A remaining duplicate does not automatically mean it should be deleted;
# inspect the record and confirm whether it is truly redundant.

print("Demographics exact duplicates:", demographics.duplicated().sum())
print("Transactions exact duplicates:", transactions.duplicated().sum())
print("Social media exact duplicates:", social.duplicated().sum())
```

## 6.3 Identifier Integrity

```python
# =============================================================================
# SECTION 6.3: CHECK IDENTIFIER INTEGRITY
# =============================================================================
# Repeated IDs should be checked separately from exact duplicate rows.
# A repeated identifier with conflicting values may indicate a different
# data-quality issue and should not be silently removed.

if "CustomerID" in demographics.columns:
    print("Duplicated demographic CustomerIDs:",
          demographics["CustomerID"].duplicated().sum())

if "TransactionID" in transactions.columns:
    print("Duplicated TransactionIDs:",
          transactions["TransactionID"].duplicated().sum())

if "InteractionID" in social.columns:
    print("Duplicated InteractionIDs:",
          social["InteractionID"].duplicated().sum())
```

## 6.4 Date Validation

```python
# =============================================================================
# SECTION 6.4: VALIDATE DATE VARIABLES
# =============================================================================
# Convert date columns to datetime only if they are not already stored that way.
# errors="coerce" changes unparseable values to NaT so they can be detected.

date_columns = [
    (demographics, "SignupDate"),
    (transactions, "TransactionDate"),
    (social, "InteractionDate")
]

for df, col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"{col} unparseable/missing dates:", df[col].isna().sum())
```

## 6.5 Cross-Dataset CustomerID Integrity

```python
# =============================================================================
# SECTION 6.5: CHECK CUSTOMER IDs ACROSS DATASETS
# =============================================================================
# Transaction and social-media customer IDs should correspond to customers
# represented in the demographics dataset.

demographic_ids = set(demographics["CustomerID"].dropna())
transaction_ids = set(transactions["CustomerID"].dropna())
social_ids = set(social["CustomerID"].dropna())

print("Transaction CustomerIDs not found in demographics:",
      len(transaction_ids - demographic_ids))

print("Social CustomerIDs not found in demographics:",
      len(social_ids - demographic_ids))
```

---

# 7. Descriptive Statistics

# 7.1 Measures of Central Tendency

## Customer Age

```python
# =============================================================================
# SECTION 7.1A: CENTRAL TENDENCY OF CUSTOMER AGE
# =============================================================================
# Mean represents the overall average age.
# Median represents the middle age after sorting all valid observations.
# Mode identifies the most frequently occurring age.
# Compare mean and median to help identify whether the distribution may be
# relatively balanced or influenced by values concentrated on one side.

age = demographics["Age"].dropna()

age_mean = age.mean()
age_median = age.median()
age_mode = age.mode()

print(f"Mean Age: {age_mean:.2f}")
print(f"Median Age: {age_median:.2f}")
print("Mode Age(s):", age_mode.tolist())
```

## Transaction Amount

```python
# =============================================================================
# SECTION 7.1B: CENTRAL TENDENCY OF TRANSACTION AMOUNT
# =============================================================================
# Mean measures the average transaction amount.
# Median represents the middle valid transaction amount.
# Comparing mean and median helps determine whether unusually high or low
# values may influence the average.

amount = transactions["Amount"].dropna()

amount_mean = amount.mean()
amount_median = amount.median()
amount_mode = amount.mode()

print(f"Mean Transaction Amount: {amount_mean:.2f}")
print(f"Median Transaction Amount: {amount_median:.2f}")
print("Mode Transaction Amount(s):", amount_mode.tolist()[:10])
```

---

# 8. Measures of Variability

## 8.1 Age Variability

```python
# =============================================================================
# SECTION 8.1: VARIABILITY OF CUSTOMER AGE
# =============================================================================
# Minimum and maximum identify the observed extremes.
# Range measures the full distance between those extremes.
# Variance measures average squared deviation from the mean.
# Standard deviation describes the typical spread around the mean.

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

## 8.2 Transaction Amount Variability

```python
# =============================================================================
# SECTION 8.2: VARIABILITY OF TRANSACTION AMOUNT
# =============================================================================
# A larger standard deviation means transaction amounts are more dispersed
# around the mean. It does not automatically prove instability or risk.

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

---

# 9. Measures of Position, Quartiles, and IQR

## 9.1 Transaction Amount Five-Number Summary

```python
# =============================================================================
# SECTION 9.1: QUARTILES AND IQR FOR TRANSACTION AMOUNT
# =============================================================================
# Q1 marks the 25th percentile.
# Q2 is the median or 50th percentile.
# Q3 marks the 75th percentile.
# IQR measures the spread of the middle 50% of valid transaction values.

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

## 9.2 Detect Potential Statistical Outliers

```python
# =============================================================================
# SECTION 9.2: IDENTIFY POTENTIAL TRANSACTION-AMOUNT OUTLIERS
# =============================================================================
# The IQR rule identifies observations that deserve investigation.
# Values outside the bounds should not be automatically deleted because a
# statistically unusual transaction may still be a valid business event.

amount_outliers = transactions[
    (transactions["Amount"] < lower_bound) |
    (transactions["Amount"] > upper_bound)
].copy()

print("Potential transaction-amount outliers:", len(amount_outliers))
display(amount_outliers.head(20))
```

---

# 10. Exploratory Visualizations

These figures support Milestone 1 interpretation. They are exploratory rather than the final polished visual storytelling required for Milestone 2.

## 10.1 Age Distribution

```python
# =============================================================================
# SECTION 10.1: VISUALIZE CUSTOMER AGE DISTRIBUTION
# =============================================================================
# A histogram shows where customer ages are concentrated.
# The KDE curve provides an additional view of the distribution shape.
# Use this figure together with mean, median, and standard deviation.

plt.figure(figsize=(9, 5))
sns.histplot(data=demographics, x="Age", bins=15, kde=True)
plt.axvline(age_mean, linestyle="--", label=f"Mean = {age_mean:.2f}")
plt.axvline(age_median, linestyle=":", label=f"Median = {age_median:.2f}")
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

## 10.2 Transaction Amount Distribution

```python
# =============================================================================
# SECTION 10.2: VISUALIZE TRANSACTION AMOUNT DISTRIBUTION
# =============================================================================
# This histogram shows how transaction values are distributed.
# Compare the shape with the calculated mean and median before describing
# the distribution as symmetric, skewed, or concentrated.

plt.figure(figsize=(9, 5))
sns.histplot(data=transactions, x="Amount", bins=20, kde=True)
plt.axvline(amount_mean, linestyle="--",
            label=f"Mean = {amount_mean:.2f}")
plt.axvline(amount_median, linestyle=":",
            label=f"Median = {amount_median:.2f}")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

## 10.3 Transaction Amount Box Plot

```python
# =============================================================================
# SECTION 10.3: VISUALIZE TRANSACTION AMOUNT AND POTENTIAL OUTLIERS
# =============================================================================
# A box plot summarizes the median, quartiles, spread, and possible outliers.
# Points beyond the whiskers should be investigated rather than automatically
# classified as invalid.

plt.figure(figsize=(9, 4))
sns.boxplot(data=transactions, x="Amount")
plt.title("Box Plot of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.show()
```

---

# 11. Frequency Distribution Analysis

## 11.1 Helper Function for Categorical Frequencies

```python
# =============================================================================
# SECTION 11.1: CREATE A REUSABLE CATEGORY-SUMMARY FUNCTION
# =============================================================================
# This function reports both count and percentage.
# Percentages help determine whether a category is dominant or underrepresented.

def category_summary(df, column):
    counts = df[column].value_counts(dropna=False)
    percentages = df[column].value_counts(
        dropna=False, normalize=True
    ).mul(100)

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    return summary

# Example:
display(category_summary(transactions, "ProductCategory"))
```

## 11.2 Review Major Categorical Variables

```python
# =============================================================================
# SECTION 11.2: REVIEW FREQUENCY DISTRIBUTIONS
# =============================================================================
# Review each category separately.
# Do not interpret the most frequent category as customer preference unless
# the dataset contains evidence that directly measures preference.

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
        print(f"\n--- {column} ---")
        display(category_summary(df, column))
```

## 11.3 Product Category Frequency Visualization

```python
# =============================================================================
# SECTION 11.3: VISUALIZE PRODUCT CATEGORY FREQUENCY
# =============================================================================
# A count plot makes differences in category frequency easier to compare.
# This visualization supports analysis of dominance or imbalance.

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

## 11.4 Social Platform Frequency Visualization

```python
# =============================================================================
# SECTION 11.4: VISUALIZE SOCIAL PLATFORM FREQUENCY
# =============================================================================
# This plot shows which platforms account for more recorded interactions.
# Higher frequency means more observations in the dataset, not necessarily
# stronger customer preference or better platform performance.

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

---

# 12. Grouped Data Analysis

## 12.1 Product Category vs Transaction Amount

```python
# =============================================================================
# SECTION 12.1: COMPARE TRANSACTION AMOUNTS BY PRODUCT CATEGORY
# =============================================================================
# Grouped analysis reveals differences hidden by the overall average.
# Count provides context because a high mean based on very few observations
# should be interpreted more cautiously.

product_amount_summary = (
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

display(product_amount_summary)
```

## 12.2 Product Category Comparison Visualization

```python
# =============================================================================
# SECTION 12.2: VISUALIZE MEDIAN TRANSACTION AMOUNT BY PRODUCT CATEGORY
# =============================================================================
# Median is used here because it provides a robust comparison of typical
# transaction values across categories.
# The result should be interpreted together with count and variability.

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

## 12.3 Payment Method vs Transaction Amount

```python
# =============================================================================
# SECTION 12.3: COMPARE TRANSACTION AMOUNTS BY PAYMENT METHOD
# =============================================================================
# This analysis determines whether typical transaction values and variability
# differ across recorded payment methods.

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
    .sort_values("Mean", ascending=False)
)

display(payment_summary)
```

## 12.4 Platform vs Sentiment

```python
# =============================================================================
# SECTION 12.4: EXAMINE SENTIMENT DISTRIBUTION WITHIN EACH PLATFORM
# =============================================================================
# normalize="index" converts each platform row into percentages.
# This makes platforms with different interaction counts easier to compare.

platform_sentiment = pd.crosstab(
    social["Platform"],
    social["Sentiment"],
    normalize="index"
).mul(100).round(2)

display(platform_sentiment)
```

## 12.5 Platform-Sentiment Visualization

```python
# =============================================================================
# SECTION 12.5: VISUALIZE SENTIMENT COMPOSITION BY PLATFORM
# =============================================================================
# A 100% stacked bar chart shows how sentiment composition differs by platform.
# This supports comparison of proportions rather than raw interaction counts.

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

---

# 13. Temporal Trend Analysis

## 13.1 Monthly Transaction Activity

```python
# =============================================================================
# SECTION 13.1: ANALYZE MONTHLY TRANSACTION ACTIVITY
# =============================================================================
# Monthly aggregation can reveal recurring changes or trends over time.
# Count shows transaction activity while sum shows total recorded transaction
# value. Interpret both because a high total can come from either more
# transactions or larger transaction amounts.

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
```

## 13.2 Monthly Transaction Count Visualization

```python
# =============================================================================
# SECTION 13.2: VISUALIZE MONTHLY TRANSACTION COUNT
# =============================================================================
# A line plot is appropriate because month is ordered over time.
# Describe only patterns visible in the available observation period.

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

## 13.3 Monthly Social Interaction Activity

```python
# =============================================================================
# SECTION 13.3: ANALYZE AND VISUALIZE MONTHLY SOCIAL INTERACTIONS
# =============================================================================
# This allows transaction and social-media activity to be examined over time
# without assuming that one causes the other.

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

---

# 14. Customer-Level Cross-Dataset Analysis

Raw transactions and raw social interactions should **not** be directly joined to each other by `CustomerID` because customers may have multiple records in both tables. A direct event-to-event join could create a many-to-many relationship and inflate totals.

Aggregate each event table to customer level first.

## 14.1 Customer-Level Transaction Features

```python
# =============================================================================
# SECTION 14.1: AGGREGATE TRANSACTIONS TO CUSTOMER LEVEL
# =============================================================================
# These derived features preserve customer purchasing behavior without
# multiplying transaction records during later cross-dataset joins.

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

## 14.2 Customer-Level Social Features

```python
# =============================================================================
# SECTION 14.2: AGGREGATE SOCIAL ACTIVITY TO CUSTOMER LEVEL
# =============================================================================
# InteractionCount summarizes how many recorded social interactions each
# customer has in the available dataset.

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

## 14.3 Build the Customer Analysis Table

```python
# =============================================================================
# SECTION 14.3: COMBINE CUSTOMER-LEVEL DATA
# =============================================================================
# Start with demographics because it contains one logical customer record.
# Left joins preserve demographic customers even when they have no transaction
# or social-media activity in the observation period.

customer_analysis = (
    demographics
    .merge(customer_transactions, on="CustomerID", how="left")
    .merge(customer_social, on="CustomerID", how="left")
)

# Customers with no recorded transactions/interactions have counts of zero.
# Spending remains missing if no valid transaction amount exists.
customer_analysis["TransactionCount"] = (
    customer_analysis["TransactionCount"].fillna(0)
)
customer_analysis["InteractionCount"] = (
    customer_analysis["InteractionCount"].fillna(0)
)

display(customer_analysis.head())
print("Customer-level analytical shape:", customer_analysis.shape)
```

---

# 15. Cross-Dataset Grouped Analysis

## 15.1 Income Level vs Customer Spending

```python
# =============================================================================
# SECTION 15.1: COMPARE CUSTOMER SPENDING BY INCOME LEVEL
# =============================================================================
# Compare mean and median because group-level spending may be affected by
# customers with unusually high total spending.
# Count should always be reviewed to understand the size of each group.

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
```

## 15.2 Income-Level Spending Visualization

```python
# =============================================================================
# SECTION 15.2: VISUALIZE CUSTOMER SPENDING BY INCOME LEVEL
# =============================================================================
# A box plot shows the distribution of TotalSpend inside each income group,
# including median, spread, and possible high/low observations.

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

## 15.3 Age vs Customer Spending

```python
# =============================================================================
# SECTION 15.3: EXAMINE AGE AND TOTAL SPENDING
# =============================================================================
# A scatter plot is appropriate because both variables are numerical.
# A visible pattern may indicate association, but it does not prove that age
# causes customers to spend more or less.

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

# Pearson correlation measures linear association between the two variables.
# Interpret magnitude and direction cautiously, especially if the relationship
# is nonlinear or affected by missing values/outliers.
age_spend_corr = customer_analysis[["Age", "TotalSpend"]].corr().iloc[0, 1]
print(f"Age vs TotalSpend Pearson correlation: {age_spend_corr:.3f}")
```

## 15.4 Social Engagement vs Purchasing Activity

```python
# =============================================================================
# SECTION 15.4: EXAMINE SOCIAL ENGAGEMENT AND TRANSACTION ACTIVITY
# =============================================================================
# This tests whether customers with more recorded social interactions also
# tend to have more recorded transactions.
# An association does not establish that social engagement causes purchases.

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
    "InteractionCount vs TransactionCount Pearson correlation:",
    f"{engagement_transaction_corr:.3f}"
)
```

---

# 16. Correlation Review for Numerical Customer Features

```python
# =============================================================================
# SECTION 16: REVIEW NUMERICAL CORRELATIONS
# =============================================================================
# Correlation helps identify linear relationships among numerical variables.
# Identifier fields must not be included because numerical-looking IDs do not
# represent measurable quantities.
# Correlation describes association, not causation.

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

# 17. Pattern, Trend, and Anomaly Interpretation

For each major result, use the following reasoning sequence:

**Business Question → Analytical Method → Result → Interpretation**

Then document important findings using:

**Finding → Evidence → Interpretation → Business Implication**

Example structure:

```text
Finding:
[State the observed result.]

Evidence:
[Provide the actual statistic, frequency, grouped comparison, trend, or chart.]

Interpretation:
[Explain what the result means within the FinMark dataset.]

Business Implication:
[Explain why FinMark may need to investigate or visualize the pattern further.]

Limitation:
[State what the evidence does not prove.]
```

Do not fill this section with assumptions. Insert only findings directly supported by the notebook outputs.

---

# 18. Decision-Oriented Analysis

For the strongest findings, answer the following five questions.

## 18.1 What is the most important finding?

State the strongest evidence-supported observation from the EDA.

## 18.2 What insight can be derived?

Explain what the finding suggests without claiming unsupported causation.

## 18.3 What additional information should be investigated?

Possible follow-up questions may include:

- What explains differences in purchasing behavior between customer groups?
- Are unusual transaction amounts legitimate refunds, promotions, or errors?
- Are missing or underrepresented categories affecting the observed patterns?
- Does a longer observation period show the same trends?
- Are additional customer preference or marketing variables needed?

## 18.4 What action should be recommended?

Recommendations may include:

- prioritize specific patterns for Milestone 2 visualization;
- investigate data-quality issues that remain unresolved;
- evaluate useful derived variables as possible future ML features;
- obtain additional data needed to test possible explanations.

Do not recommend operational changes solely from correlation.

## 18.5 How should success be measured?

Potential future measures include:

- transaction frequency;
- average and median spending;
- total customer spending;
- repeat transaction activity if derivable;
- social interaction frequency;
- sentiment distribution;
- changes across customer segments; and
- model performance metrics once an ML objective has been formally defined.

Do not invent arbitrary improvement targets unless FinMark provides a business baseline or requirement.

---

# 19. Machine Learning Readiness Assessment

Review the following before declaring the data ready for future modeling.

## 19.1 Data Quality

- Are important variables sufficiently complete?
- Are duplicate records resolved or documented?
- Are data types appropriate?
- Do questionable values remain?
- Are dates usable for temporal feature engineering?

## 19.2 Feature Quality

Possible useful features include:

- Age
- IncomeLevel
- ProductCategory
- PaymentMethod
- transaction amount
- TransactionCount
- TotalSpend
- AverageSpend
- InteractionCount
- validated sentiment categories
- derived temporal variables

Identifiers such as `CustomerID`, `TransactionID`, and `InteractionID` should generally remain relationship keys rather than ordinary predictive numerical features.

## 19.3 Data Balance

Review categorical distributions to determine whether some groups are strongly underrepresented.

Imbalance is not automatically a data-quality failure, but it may influence future:

- training strategy;
- validation strategy;
- class weighting or resampling decisions; and
- model evaluation metrics.

## 19.4 Outlier Sensitivity

Potential statistical outliers should be investigated before deciding whether:

- they are valid observations;
- transformation is appropriate;
- scaling is needed;
- robust statistical/modeling methods are preferable; or
- exclusion is justified.

## 19.5 Remaining Risks

Document any unresolved issues that could affect future model development.

---

# 20. Key Findings and Recommendations

Summarize only findings already established by the analysis.

Recommended structure:

## 20.1 Data Quality Findings
- [Evidence-supported finding]
- [Evidence-supported finding]

## 20.2 Statistical Findings
- [Evidence-supported finding]
- [Evidence-supported finding]

## 20.3 Behavioral Patterns
- [Evidence-supported finding]

## 20.4 Important Relationships
- [Evidence-supported finding]

## 20.5 Remaining Issues
- [Limitation or unresolved data issue]

## 20.6 Recommendations
- [Milestone 2 visualization priority]
- [Further investigation]
- [Potential future ML consideration]

---

# 21. Limitations

Potential limitations to discuss when supported by the final analysis:

- EDA identifies associations but does not establish causality.
- Missing data may reduce analytical coverage.
- Some unusual transactions may lack sufficient business context for definitive classification.
- Category imbalance can affect comparisons.
- Customer-level aggregation changes the unit of analysis.
- Available variables may not explain why observed customer behaviors occur.
- Findings apply to the supplied FinMark datasets and should not automatically be generalized beyond them.

---

# 22. Conclusion

The conclusion should answer:

1. What did the EDA reveal about the structure and quality of the FinMark datasets?
2. What were the most important statistical characteristics?
3. What patterns, relationships, or anomalies deserve further investigation?
4. Are the datasets sufficiently prepared for Milestone 2?
5. What considerations should be carried forward into future machine learning activities?

Do not introduce new evidence in the conclusion.

---

# 23. Mapping to the Six AI-Graded Questions

| AI-Graded Question | Main Evidence / Report Sections |
|---|---|
| **Q1. Describe the datasets used in EDA** | Sections 1–6 |
| **Q2. Discuss preprocessing procedures** | Section 6 and Week 2–3 preprocessing records |
| **Q3. Discuss descriptive statistics** | Sections 7–10 |
| **Q4. Analyze frequency and value distributions** | Sections 9–13 |
| **Q5. Discuss patterns, relationships, trends, anomalies, or outliers** | Sections 12–17 |
| **Q6. Summarize findings and recommendations** | Sections 18–22 |

---

# 24. Final Analytical Chain

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
QUARTILES / IQR / DISTRIBUTION
        ↓
FREQUENCY ANALYSIS
        ↓
GROUPED ANALYSIS
        ↓
TEMPORAL ANALYSIS
        ↓
CUSTOMER-LEVEL CROSS-DATASET ANALYSIS
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
FURTHER INVESTIGATION
        ↓
RECOMMENDATION
        ↓
SUCCESS MEASURE
        ↓
MACHINE LEARNING READINESS
        ↓
MILESTONE 2 DATAVIZ
```

---

# 25. Final Review Checklist

Before submission, verify that:

- [ ] The FinMark business problem is clearly defined.
- [ ] All three datasets are described.
- [ ] Numerical and categorical variables are identified.
- [ ] Week 2–3 preprocessing decisions are explained.
- [ ] Duplicate handling is documented.
- [ ] Missing-value handling is documented.
- [ ] Data-type corrections are documented.
- [ ] Post-cleaning validation is shown.
- [ ] Mean is calculated where appropriate.
- [ ] Median is calculated where appropriate.
- [ ] Mode is calculated where appropriate.
- [ ] Minimum and maximum are reviewed.
- [ ] Range is calculated.
- [ ] Variance is calculated.
- [ ] Standard deviation is calculated.
- [ ] Quartiles and IQR are reviewed.
- [ ] Frequency distributions are analyzed.
- [ ] Data balance or imbalance is discussed.
- [ ] Potential outliers are investigated.
- [ ] Grouped analysis is included.
- [ ] Temporal trends are examined where applicable.
- [ ] Cross-dataset joins preserve the correct analytical grain.
- [ ] Relationship analysis avoids causal claims.
- [ ] Exploratory visualizations support important statistical findings.
- [ ] Every major finding has evidence.
- [ ] Findings are interpreted in FinMark's business context.
- [ ] Limitations are documented.
- [ ] Future ML considerations are evidence-based.
- [ ] Recommendations do not exceed what the data supports.
- [ ] Questions 1–6 can be answered directly from notebook evidence.
- [ ] The final report uses professional and academic writing.
