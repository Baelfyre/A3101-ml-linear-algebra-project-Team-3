# FinMark Corporation: ML Solution Dataset Preprocessing

**Course:** MO-IT162 - Math for Machine Learning: Linear Algebra  
**Section:** A3101  
**Team:** Team 3  
**Activity:** Week 3 - ML Solution Datasets Preprocessing

## Team Members

- Ghaylord Benedict Lugod
- James Lynelle Ongo
- Kaselyn Cates Ponteres
- Mark Anthony Goyon

---

## Project Overview

FinMark Corporation aims to improve its ability to analyze customer data, identify meaningful patterns, and generate useful business insights through a machine learning solution.

The FinMark project is completed through three major stages:

1. **Milestone 1: ML Solution EDA**  
   Explore the customer demographics, transaction, and social media datasets to understand their structure, identify patterns, detect anomalies, and assess data quality.

2. **Milestone 2: ML Solution DataViz**  
   Develop visualizations that reveal relationships, trends, and patterns related to customer behavior and market activity.

3. **Terminal Assessment: ML Solution Presentation**  
   Present the analytical process, visualizations, challenges, findings, and proposed machine learning solution.

This Week 3 activity focuses on **dataset preprocessing in preparation for Milestone 1**. The notebook verifies the supplied raw datasets, documents data-quality issues before modification, applies justified cleaning procedures, validates the cleaned outputs, and exports finalized preprocessing files.

The notebook serves as the detailed and reproducible technical record of the preprocessing workflow. This README provides a concise summary of the problem, datasets, findings, cleaning decisions, results, limitations, and Week 3 submission outputs.

**Google Colab:** [Open the Week 3 ML Solution Dataset Preprocessing Notebook](https://colab.research.google.com/drive/1lbjazGCL2ae6J_jf16GifIwk81prpxle?usp=drive_link)
---

## Business Problem

FinMark Corporation collects customer-related information from multiple sources, including customer demographics, transaction records, and social media interactions.

However, incomplete, duplicated, incorrectly formatted, conflicting, or analytically invalid values can reduce the reliability of later exploratory analysis and machine learning results.

Before meaningful EDA can be performed, the datasets must therefore be verified and cleaned so that the information used for analysis is sufficiently consistent, accurate, traceable, and usable.

The preprocessing problem is not simply to make the datasets appear clean. The objective is to establish a defensible analytical dataset while preserving valid information and avoiding unsupported assumptions.

---

## Week 3 Objectives

The Week 3 preprocessing activity aims to:

- collect and load the required FinMark datasets;
- preserve the original raw datasets;
- verify dataset dimensions, columns, data types, and contents;
- identify and document missing values;
- verify exact duplicate records;
- investigate duplicate and conflicting identifiers;
- validate numeric fields and questionable values;
- investigate date-format inconsistencies;
- review categorical consistency;
- verify cross-dataset `CustomerID` relationships;
- review the analytical relevance of available variables;
- define cleaning rules before modifying data;
- create separate cleaned working datasets;
- apply justified cleaning procedures;
- validate the cleaned datasets;
- compare raw and cleaned dataset sizes;
- export finalized cleaned CSV files; and
- reload and validate the actual exported Week 3 outputs.

---

## Scope

### Included

This preprocessing workflow covers:

- data collection and loading;
- structural verification;
- missing-value verification;
- duplicate verification;
- identifier integrity checks;
- numeric validation;
- date-format validation;
- categorical consistency checks;
- variable relevance review;
- cleaning decision documentation;
- data cleaning;
- post-cleaning validation;
- before-versus-after comparison;
- CSV export; and
- exported-file validation.

### Outside the Scope

The following are not performed in this Week 3 notebook:

- machine learning model development;
- model training or evaluation;
- final feature engineering for a specific model;
- final data visualization;
- causal analysis; and
- final business recommendations based on Milestone 1 EDA.

These activities belong to later stages of the FinMark project.

---

## Data Collection

The project uses only the official FinMark MFML datasets supplied for the course. No external dataset was introduced during this preprocessing activity.

The source files are:

- `customer_demographics_contaminated.csv`
- `customer_transactions_contaminated.csv`
- `social_media_interactions_contaminated.csv`

The raw datasets are stored separately from the generated preprocessing outputs so that the original files remain unchanged.

Each raw dataset initially contains:

- **3,200 records**
- **6 variables**

---

## Dataset Roles

| Dataset | Purpose |
| --- | --- |
| **Customer Demographics** | Provides customer-level attributes for customer profiling, segmentation, socioeconomic analysis, geographic analysis, and signup timing |
| **Customer Transactions** | Provides transaction-level information for spending analysis, product preferences, payment behavior, and transaction timing |
| **Social Media Interactions** | Provides interaction-level information for engagement analysis, platform usage, interaction behavior, sentiment, and interaction timing |

`CustomerID` is the common customer identifier used to relate the three datasets.

`TransactionID` and `InteractionID` identify individual transaction and social media interaction records.

---

## Variables Used

### Customer Demographics

- `CustomerID`
- `Age`
- `Gender`
- `Location`
- `IncomeLevel`
- `SignupDate`

### Customer Transactions

- `CustomerID`
- `TransactionID`
- `TransactionDate`
- `Amount`
- `ProductCategory`
- `PaymentMethod`

### Social Media Interactions

- `CustomerID`
- `InteractionID`
- `InteractionDate`
- `Platform`
- `InteractionType`
- `Sentiment`

All available fields were reviewed for relevance before considering removal.

Identifier fields are retained for traceability, relationships, grouping, duplicate checking, and integrity verification, but they are not intended to be treated as ordinary machine learning features.

---

## Preprocessing Principles

The workflow follows these principles:

1. Preserve the original FinMark datasets.
2. Identify and document data-quality issues before cleaning.
3. Apply changes only to separate cleaned working copies.
4. Avoid unsupported imputation or estimated replacement values.
5. Remove records only when a documented and defensible rule supports removal.
6. Standardize valid information when the problem is formatting rather than meaning.
7. Investigate duplicate and conflicting records before treatment.
8. Validate questionable numeric values before classifying them as invalid.
9. Preserve legitimate categorical values unless evidence supports normalization or consolidation.
10. Retain identifiers for relational and integrity purposes.
11. Measure the effect of preprocessing through before-versus-after comparisons.
12. Validate both the cleaned DataFrames and the actual exported files.

---

# Data Verification Findings

## Missing Values

The raw datasets contain missing values in the following variables:

| Dataset | Variable | Missing Values |
| --- | --- | ---: |
| Customer Demographics | `Age` | 291 |
| Customer Demographics | `IncomeLevel` | 303 |
| Customer Transactions | `Amount` | 304 |
| Customer Transactions | `ProductCategory` | 299 |
| Social Media Interactions | `Platform` | 311 |
| Social Media Interactions | `Sentiment` | 329 |

The remaining variables contained no null values during the initial missing-value verification.

### Missing-Data Decision

Missing information is not imputed because the original values cannot be reliably reconstructed from the supplied FinMark datasets.

Rows containing missing values are therefore omitted from the cleaned datasets.

This reduces the available sample and may affect the representation of some customer groups or behaviors. The final row reductions are documented in the preprocessing results.

---

## Exact Duplicate Records

The raw datasets contain confirmed exact duplicate rows:

| Dataset | Exact Duplicate Rows |
| --- | ---: |
| Customer Demographics | 177 |
| Customer Transactions | 185 |
| Social Media Interactions | 180 |

Exact duplicate records are removed because identical records can overrepresent the same observation during later analysis.

Repeated `CustomerID` values in the transaction and social media datasets are not automatically treated as duplicate errors because one customer may legitimately have multiple transactions or social interactions.

---

## Conflicting Record Identifiers

Exact duplicate checking alone does not establish identifier integrity. Record-level identifiers expected to identify one record were therefore checked for non-identical duplicates.

The workflow verified:

- `CustomerID` in Customer Demographics;
- `TransactionID` in Customer Transactions; and
- `InteractionID` in Social Media Interactions.

Where the same record-level identifier appeared in multiple non-identical records, no authoritative field was available to determine which version was correct.

### Cleaning Decision

All records belonging to an unresolved conflicting identifier group are omitted rather than arbitrarily retaining the first occurrence.

This prevents row order from being used as an unsupported basis for deciding which conflicting record is authoritative.

---

## Numeric Verification

### Customer Age

`Age` was initially imported as an object field and therefore required validation before numeric conversion.

The verification identified:

- **291 missing Age values**
- **34 non-numeric Age records**
- **63 questionable numeric Age records**

The non-numeric value identified was:

- `Unknown`

The questionable numeric values were:

- `-1`
- `150`

These conditions were treated separately because they represent different problems:

- a blank value is missing information;
- `Unknown` is present but cannot provide a valid numeric age;
- `-1` and `150` are numeric but fall outside the documented analytical validity range.

### Age Cleaning Decision

- `Unknown` values are omitted.
- Ages below `0` are omitted.
- Ages above `120` are omitted.
- Valid ages are converted to numeric values.

Valid observed customer ages range from 18 to 70 years. The values of -1 and 180 were treated as invalid because they fall outside the observed and plausible customer age range.

---

## Transaction Amount Verification

The `Amount` field was also initially imported as an object and required numeric validation.

The verification identified:

- **304 missing Amount values**
- **30 non-numeric Amount records**
- **40 negative Amount records**
- **30 zero-value Amount records**

The non-numeric value identified was:

- `Free`

The negative transaction amount identified during the review was:

- `-100`

### Amount Cleaning Decision

- Missing Amount values are handled under the documented missing-value policy.
- `Free` is converted to `0.0`.
- Negative Amount values are omitted.
- Existing numeric zero values are retained.

`Free` is treated as semantic normalization rather than imputation because the value explicitly indicates no monetary charge.

Negative values are omitted because the supplied dataset does not contain a transaction-type field establishing whether the value represents a legitimate refund, reversal, credit, or another negative transaction event.

Zero is retained because it is numerically valid and there is insufficient evidence that a zero-value transaction is erroneous.

---

## Date-Format Verification

The three date variables are:

- `SignupDate`
- `TransactionDate`
- `InteractionDate`

Initial temporary date conversion reported:

| Date Field | Initial Parsing Failures |
| --- | ---: |
| `SignupDate` | 98 |
| `TransactionDate` | 98 |
| `InteractionDate` | 1,986 |

The affected values were investigated before any records were removed.

The datasets were found to contain two valid date representations:

- `YYYY-MM-DD`
- `DD/MM/YYYY`

After explicitly parsing the verified formats:

| Date Field | Remaining Parsing Failures |
| --- | ---: |
| `SignupDate` | 0 |
| `TransactionDate` | 0 |
| `InteractionDate` | 0 |

### Date Cleaning Decision

The initial failures are treated as **format inconsistencies, not invalid dates**.

No record is removed because of the mixed date formats.

All valid date values are normalized and exported using:

```text
YYYY-MM-DD
```

---

## Categorical Consistency Verification

The following categorical variables were reviewed:

### Customer Demographics

- `Gender`
- `IncomeLevel`
- `Location`

### Customer Transactions

- `ProductCategory`
- `PaymentMethod`

### Social Media Interactions

- `Platform`
- `InteractionType`
- `Sentiment`

Temporary lowercase and whitespace normalization produced:

**0 possible capitalization or surrounding-whitespace category differences.**

This indicates that the existing category labels were already consistently formatted.

### Sentiment Categories

The five non-missing sentiment categories were preserved:

- Positive
- Negative
- Neutral
- Very Positive
- Very Negative

No evidence supported combining:

- `Positive` with `Very Positive`; or
- `Negative` with `Very Negative`.

### Location

`Location` contained **2,696 distinct values**.

The field was retained because high cardinality alone does not make a variable invalid or irrelevant. It may require additional treatment during later EDA or model preparation depending on the analytical question.

---

## Cross-Dataset CustomerID Integrity

`CustomerID` was checked across the three raw datasets.

### Unique Customer Coverage

| Dataset | Unique CustomerIDs |
| --- | ---: |
| Customer Demographics | 3,000 |
| Customer Transactions | 1,871 |
| Social Media Interactions | 1,893 |

### Orphan CustomerIDs

- Transaction CustomerIDs not found in raw demographics: **0**
- Social Media CustomerIDs not found in raw demographics: **0**

This confirms that all activity CustomerIDs originally referenced customers represented in the raw Customer Demographics dataset.

### Customers Without Recorded Activity

- Demographic customers without transactions: **1,129**
- Demographic customers without social media interactions: **1,107**

These cases are not automatically treated as data-quality errors because a valid customer may have no recorded transaction or social media activity during the available observation period.

---

## Data Relevance Review

All 18 dataset fields were reviewed for their role in the FinMark project.

The fields provide information about:

- customer characteristics;
- customer registration timing;
- geographic information;
- socioeconomic characteristics;
- purchasing activity;
- transaction value;
- product preferences;
- payment behavior;
- social media engagement;
- platform usage;
- interaction type; and
- sentiment.

No variable was removed solely because of high cardinality or because it was not intended as a model feature.

Identifier fields remain in the cleaned master datasets because they support:

- record identification;
- relationships;
- grouping;
- duplicate checking;
- integrity verification; and
- traceability.

---

# Preprocessing Approach

The preprocessing process uses a verification-first workflow:

```text
Raw FinMark Datasets
        |
        v
Structural Verification
        |
        v
Data-Quality Verification
        |
        v
Investigate Questionable Records
        |
        v
Document Findings
        |
        v
Define Cleaning Rules
        |
        v
Create Separate Cleaned Working Copies
        |
        v
Apply Cleaning
        |
        v
Post-Cleaning Validation
        |
        v
Before-vs-After Comparison
        |
        v
Export Final CSV Files
        |
        v
Reload and Validate Exported Outputs
```

This approach prevents cleaning decisions from being made solely because a value appears unusual or because a parser fails.

---

# Cleaning Plan and Justification

| Data-Quality Condition | Treatment | Justification |
| --- | --- | --- |
| Exact duplicate rows | Remove | Prevents identical observations from being overrepresented |
| Unresolved conflicting identifiers | Remove the entire conflicting identifier group | No authoritative field identifies which conflicting record is correct |
| Missing values | Omit affected records | Missing information cannot be reliably reconstructed without unsupported imputation |
| `Age = Unknown` | Omit affected record | No valid numeric age can be recovered |
| Age below 0 or above 120 | Omit affected record | Values fall outside the documented analytical validity range |
| `Amount = Free` | Convert to `0.0` | Explicitly represents no monetary charge |
| Negative Amount | Omit affected record | No transaction-type field supports interpreting the negative value as a legitimate refund, reversal, or credit |
| Numeric zero Amount | Retain | Zero is valid numeric data and is not proven erroneous |
| Mixed valid date formats | Normalize to `YYYY-MM-DD` | Explicit parsing confirmed the dates were valid but inconsistently represented |
| Categorical labels | Preserve | No capitalization or whitespace-based duplicates were detected |
| Column names | Convert to `snake_case` | Improves consistency and readability for later Pandas analysis |
| Identifier fields | Retain | Required for relationships, traceability, grouping, duplicate checking, and integrity |

---

# Data Cleaning

Cleaning was applied only to separate `_clean` DataFrames. The original raw DataFrames were preserved.

The main cleaning operations were:

1. Create clean working copies.
2. Initialize a cleaning audit log.
3. Normalize column names to `snake_case`.
4. Remove confirmed exact duplicate rows.
5. Remove unresolved conflicting identifier groups.
6. Omit rows containing missing values.
7. Convert and validate customer Age.
8. Normalize `Free` Amount values to `0.0`.
9. Remove unsupported negative Amount records.
10. Retain valid zero-value transactions.
11. Normalize date fields using the verified mixed-format parser.
12. Reset DataFrame indexes.
13. Validate the cleaned datasets.

---

## Cleaning Audit

Row-removal counts are sequential.

A record may satisfy more than one raw data-quality condition. Once a record is removed by an earlier cleaning operation, it cannot be removed again during a later operation.

For this reason, raw issue counts should not be added together to calculate the total number of removed records.

| Dataset | Cleaning Step | Rows Before | Rows After | Rows Removed |
| --- | --- | ---: | ---: | ---: |
| Customer Demographics | Remove exact duplicate rows | 3,200 | 3,023 | 177 |
| Customer Transactions | Remove exact duplicate rows | 3,200 | 3,015 | 185 |
| Social Media Interactions | Remove exact duplicate rows | 3,200 | 3,020 | 180 |
| Customer Demographics | Remove unresolved `CustomerID` conflicts | 3,023 | 2,977 | 46 |
| Customer Transactions | Remove unresolved `TransactionID` conflicts | 3,015 | 2,985 | 30 |
| Social Media Interactions | Remove unresolved `InteractionID` conflicts | 3,020 | 2,980 | 40 |
| Customer Demographics | Omit rows containing missing values | 2,977 | 2,449 | 528 |
| Customer Transactions | Omit rows containing missing values | 2,985 | 2,452 | 533 |
| Social Media Interactions | Omit rows containing missing values | 2,980 | 2,411 | 569 |
| Customer Demographics | Remove unusable or invalid Age records | 2,449 | 2,378 | 71 |
| Customer Transactions | Normalize `Free` to `0` and remove unsupported negative Amount records | 2,452 | 2,420 | 32 |

---

# Final Preprocessing Results

The finalized preprocessing workflow produced the following dataset sizes:

| Dataset | Raw Rows | Cleaned Rows | Rows Removed | Percent Removed |
| --- | ---: | ---: | ---: | ---: |
| Customer Demographics | 3,200 | **2,378** | **822** | **25.69%** |
| Customer Transactions | 3,200 | **2,420** | **780** | **24.38%** |
| Social Media Interactions | 3,200 | **2,411** | **789** | **24.66%** |

---

## Final Cleaned Dataset Structure

### Customer Demographics

**2,378 rows**

| Variable | Final Type / Role |
| --- | --- |
| `customer_id` | Identifier |
| `age` | Numeric integer |
| `gender` | Categorical |
| `location` | Categorical / geographic |
| `income_level` | Categorical |
| `signup_date` | Date |

### Customer Transactions

**2,420 rows**

| Variable | Final Type / Role |
| --- | --- |
| `customer_id` | Customer identifier |
| `transaction_id` | Transaction identifier |
| `transaction_date` | Date |
| `amount` | Numeric float |
| `product_category` | Categorical |
| `payment_method` | Categorical |

### Social Media Interactions

**2,411 rows**

| Variable | Final Type / Role |
| --- | --- |
| `customer_id` | Customer identifier |
| `interaction_id` | Interaction identifier |
| `interaction_date` | Date |
| `platform` | Categorical |
| `interaction_type` | Categorical |
| `sentiment` | Categorical |

---

# Post-Cleaning Validation

The cleaned datasets were revalidated after preprocessing.

The validation checked:

- missing values;
- exact duplicate rows;
- record-level identifier uniqueness;
- numeric Age type;
- Age range from 0 to 120;
- numeric transaction Amount;
- non-negative transaction Amount;
- successful `SignupDate` parsing;
- successful `TransactionDate` parsing; and
- successful `InteractionDate` parsing.

Every post-cleaning validation check passed.

The notebook reported:

```text
All post-cleaning validation checks passed: True
```

---

## Post-Cleaning Cross-Dataset Coverage

The three datasets were cleaned independently.

Cleaning customer demographics can remove a customer-level record even when that customer's transaction or social media activity record is otherwise valid.

After cleaning:

| Activity Dataset | CustomerIDs Without Clean Demographics | Affected Activity Rows |
| --- | ---: | ---: |
| Customer Transactions | 336 | 492 |
| Social Media Interactions | 336 | 486 |

These activity records were retained because the corresponding CustomerIDs existed in the original raw demographics dataset and the activity records themselves were not independently shown to be invalid.

This is therefore treated as a **post-cleaning coverage limitation**, not as a new source-integrity error.

### Implication for Milestone 1

The eventual merge strategy should depend on the analytical question.

For example:

- an **inner join** would restrict the analysis to records with matching customer demographics;
- a **left join** would preserve the population of the selected base dataset.

No final merge strategy is imposed during Week 3 preprocessing.

---

# Export and Output Validation

The finalized cleaned DataFrames were exported as CSV files.

The exported files were then reloaded into Pandas so that the actual submission files could be checked rather than assuming that successful export guaranteed correctness.

The reloaded outputs showed:

| Dataset | Expected Rows | Reloaded Rows | Missing Values | Exact Duplicates | Row Count Match |
| --- | ---: | ---: | ---: | ---: | --- |
| Customer Demographics | 2,378 | 2,378 | 0 | 0 | True |
| Customer Transactions | 2,420 | 2,420 | 0 | 0 | True |
| Social Media Interactions | 2,411 | 2,411 | 0 | 0 | True |

The exported date fields were also checked for the required standardized representation:

| Date Field | `YYYY-MM-DD` Valid |
| --- | --- |
| `signup_date` | True |
| `transaction_date` | True |
| `interaction_date` | True |

The final notebook reported:

```text
Week 3 finalized output validation passed: True
```

---

# Limitations and Considerations

## Sample Reduction

The omission of incomplete, conflicting, or invalid records reduced the available sample:

- Customer Demographics: **25.69% removed**
- Customer Transactions: **24.38% removed**
- Social Media Interactions: **24.66% removed**

This may affect the representation of some customer groups or behaviors during later analysis.

## No Missing-Value Imputation

Missing values were not imputed because no defensible replacement values were available from the supplied FinMark datasets.

This avoids introducing estimated values but reduces the available sample.

## Negative Transaction Interpretation

The dataset does not contain a transaction-type field.

Negative Amount values therefore cannot be confirmed as:

- refunds;
- reversals;
- credits; or
- another legitimate negative transaction event.

The negative records were omitted under the documented Week 3 analytical rule.

## High-Cardinality Location

`Location` contains 2,696 distinct raw values.

The field remains potentially useful for geographic analysis but may require additional grouping or transformation depending on the Milestone 1 analytical question or future model design.

## Identifier Fields

`CustomerID`, `TransactionID`, and `InteractionID` remain in the cleaned datasets for relational and integrity purposes.

They should not automatically be used as ordinary predictive features in a machine learning model.

## Post-Cleaning Coverage Gaps

Independent cleaning produces activity records whose customer demographic record may no longer be available in the cleaned demographics dataset.

These coverage differences must be considered when the datasets are integrated for EDA or later machine learning work.

## Interpretation Boundaries

Week 3 preprocessing verifies and prepares data quality.

It does not establish:

- causal relationships;
- customer behavior conclusions;
- predictive findings;
- business recommendations; or
- model performance.

Those require later analysis.

---

# Week 3 Submission Outputs

The finalized Week 3 package is organized as:

```text
Week3_Preprocessing_Submission/
├── README.md
├── ml_preprocessing_workflow.ipynb
├── customer_demographics_cleaned.csv
├── customer_transactions_cleaned.csv
└── social_media_interactions_cleaned.csv
```

## `README.md`

Provides the project-level summary of:

- team information;
- project context;
- business problem;
- scope;
- data;
- initial findings;
- preprocessing approach;
- cleaning rules and justifications;
- final results;
- validation status;
- limitations; and
- submission outputs.

## `ml_preprocessing_workflow.ipynb`

Contains the complete reproducible technical workflow, including:

- project context;
- data collection;
- environment setup;
- data loading;
- structural verification;
- missing-value verification;
- duplicate verification;
- numeric verification;
- date-format investigation;
- categorical consistency verification;
- identifier integrity verification;
- data relevance review;
- pre-cleaning findings;
- cleaning decisions;
- data cleaning;
- cleaning audit;
- post-cleaning validation;
- before-versus-after comparison;
- export; and
- final exported-file validation.

## `customer_demographics_cleaned.csv`

Final cleaned customer-level dataset with **2,378 records**.

## `customer_transactions_cleaned.csv`

Final cleaned transaction-level dataset with **2,420 records**.

## `social_media_interactions_cleaned.csv`

Final cleaned social media interaction dataset with **2,411 records**.

---

# Reproducibility Notes

The notebook is designed so that:

- the raw datasets remain unchanged;
- cleaned DataFrames are created separately;
- verification occurs before treatment;
- cleaning decisions are documented;
- row-removal effects are logged;
- final results are generated from executed DataFrames;
- exported files are reloaded; and
- output files are validated before submission.

For reproducibility, the finalized notebook should be executed from the beginning in a fresh Google Colab runtime before the final version is submitted.

---

# Week 3 Submission Checklist

Before submission, verify that:

- [ ] the finalized notebook runs successfully from top to bottom;
- [ ] the three cleaned CSV files are present;
- [ ] the README reflects the final notebook results;
- [ ] no outdated or duplicate submission files remain;
- [ ] the notebook opens correctly in GitHub;
- [ ] the cleaned CSV files are accessible in GitHub;
- [ ] filenames are clear and consistent;
- [ ] the files are inside `Week3_Preprocessing_Submission`;
- [ ] the GitHub commit message clearly describes the finalized Week 3 preprocessing outputs; and
- [ ] the final repository link is ready for MyCamu submission.

---

# Handoff to Milestone 1

The Week 3 outputs provide the preprocessing foundation for **Milestone 1: ML Solution EDA**.

The next phase can use the cleaned datasets to investigate:

- customer characteristics;
- demographic distributions;
- transaction behavior;
- spending patterns;
- product preferences;
- payment behavior;
- social media engagement;
- sentiment;
- customer activity coverage;
- relationships across variables;
- anomalies; and
- meaningful trends.

Any later integration of the three datasets should explicitly document the chosen join strategy and resulting analytical population.

---

# Summary

The Week 3 FinMark preprocessing workflow established a reproducible and evidence-based path from contaminated source datasets to validated cleaned outputs.

The process first verified the raw data rather than immediately modifying questionable values. Missing information, duplicate records, conflicting identifiers, numeric anomalies, mixed date formats, categorical values, and cross-dataset relationships were investigated before treatment decisions were finalized.

The cleaning process preserved the original datasets, avoided unsupported imputation, removed records only under documented rules, normalized recoverable values, preserved legitimate categories, standardized dates, retained identifiers for integrity purposes, and recorded the effect of each cleaning step.

The final cleaned datasets contain:

- **2,378 Customer Demographics records**
- **2,420 Customer Transactions records**
- **2,411 Social Media Interaction records**

All post-cleaning validation checks passed, and the actual exported CSV files were successfully reloaded and validated with:

- matching row counts;
- zero missing values;
- zero exact duplicates; and
- standardized `YYYY-MM-DD` date fields.

The resulting Week 3 outputs are prepared for GitHub submission and provide a cleaner, more consistent, and documented foundation for **Milestone 1: ML Solution EDA**.
