# MO-IT162 Milestone 2: ML Solution Data Visualization
## Team Drafting Proposal and Notebook Skeleton

**Project:** FinMark Corporation  
**Course:** MO-IT162 Math for Machine Learning: Linear Algebra  
**Section:** A3101  
**Milestone:** Milestone 2: ML Solution DataViz  
**Environment:** Google Colab / Jupyter Notebook  
**Team:** Team 3  

> **Purpose:** This is a planning and drafting outline for the team's Milestone 2 notebook. It defines the analytical questions, intended methods, visualization choices, validation requirements, and expected interpretation structure.
>
> **Important:** This version intentionally contains no analytical answers, computed results, chart outputs, or implementation code. Team members should complete the assigned sections using the finalized cleaned FinMark datasets and evidence from Milestone 1.

---

# 1. Project Context and Business Problem

## 1.1 FinMark Problem Statement

FinMark Corporation has gathered diverse datasets, including customer transactions, social media interactions, and demographic profiles. However, the company struggles to derive actionable insights due to the volume of data and inconsistencies across sources.

These challenges hinder FinMark's ability to identify market trends and understand customer behavior, limiting data-driven decision-making and strategic planning.

## 1.2 Relationship to Milestone 1

Milestone 2 should extend the Milestone 1 EDA rather than replace or contradict it.

The intended progression is:

**Week 3 Data Preprocessing → Milestone 1 EDA → Week 7 Visualization Exploration → Milestone 2 Focused Data Visualization**

Milestone 1 established the analytical foundation by identifying distributions, patterns, trends, relationships, anomalies, limitations, and cross-dataset constraints.

Milestone 2 should:

- select the strongest findings and analytical questions from that foundation;
- visualize them using appropriate chart types;
- perform targeted follow-up analysis only when it supports the same FinMark business problem;
- clearly distinguish observations, statistical evidence, interpretation, and business implications; and
- avoid unsupported causal conclusions.

## 1.3 Milestone 2 Analytical Goal

The final notebook should use a focused set of visualizations to answer two broad questions:

1. **Market trends:** What patterns appear over time and across FinMark product activity?
2. **Customer behavior:** Which customer or engagement characteristics show meaningful relationships with purchasing behavior?

## 1.4 Stakeholders

Primary users of the analysis:

- FinMark decision-makers
- FinMark data analysts
- Marketing and customer-engagement users
- Future machine learning developers

---

# 2. Milestone 2 Objectives

The notebook should aim to:

1. visualize monthly transaction activity and identify meaningful fluctuations;
2. compare product categories using transaction activity and total transaction value;
3. describe the distribution of individual transaction amounts;
4. investigate whether recorded gender is associated with product-category participation;
5. visualize social-media sentiment composition over time;
6. investigate whether social-media engagement is associated with purchasing behavior using aligned observation periods;
7. distinguish useful analytical signals from weak or non-actionable relationships;
8. communicate findings using clear, accurate, consistent, and accessible visualizations; and
9. document analytical limitations and future machine learning considerations.

---

# 3. Scope and Boundaries

## 3.1 Included

- Customer transaction trends
- Product-category performance
- Transaction-value distribution
- Customer-level gender and product-category participation
- Social-media sentiment over time
- Customer-level social engagement and purchasing relationships
- Statistical validation where appropriate
- Evidence-based interpretation
- Business implications supported by the data

## 3.2 Excluded or Demoted from the Main Story

The following may remain in Week 7 as exploratory work but should not automatically become primary Milestone 2 visualizations:

- age versus spending scatter plots;
- income versus spending comparisons;
- payment-method versus transaction-value comparisons;
- standalone platform frequency charts;
- standalone interaction-type frequency charts;
- platform-by-sentiment comparisons;
- integrated pair plots;
- full correlation heatmaps as headline visuals;
- unsupported geographic mapping; and
- customer-tenure analysis derived from `signup_date` unless the date inconsistency is resolved.

## 3.3 Analytical Boundaries

- Do not introduce unsupported external datasets.
- Do not treat identifiers as numerical analytical features.
- Do not directly merge transaction-level rows with social-interaction-level rows when this creates many-to-many duplication.
- Aggregate event data to the customer level before cross-dataset behavioral analysis.
- Align transaction and social-media periods before direct cross-source comparisons.
- Treat statistical association as association, not causation.
- State the analytical population used for every cross-dataset result.

---

# 4. FinMark Visualization Standard

Every final chart should be reviewed against the following principles:

| Principle | Required Application |
|---|---|
| **Clarity** | One primary analytical message per visualization |
| **Relevance** | Every chart answers a FinMark business question |
| **Accuracy** | Correct units, denominators, aggregation levels, scales, and statistical methods |
| **Consistency** | Consistent labels, terminology, ordering, and formatting |
| **Accessibility** | Readable labels and interpretation that does not depend on color alone |
| **Organization** | Related visuals grouped by analytical purpose |
| **Hierarchy** | Decision-relevant findings shown before supporting diagnostics |
| **Interactivity** | Used only when it improves investigation |
| **Context** | Titles, axes, periods, populations, and limitations clearly stated |
| **Aesthetics** | Clean presentation without unnecessary decoration |

### Required reasoning chain

For each major visualization:

**Business Question → Analytical Method → Result → Visualization → Interpretation → Business Implication**

For each major finding:

**Finding → Evidence → Interpretation → Business Implication**

---

# 5. Proposed Visualization and Analysis Plan

| Section | FinMark Question | Unit of Analysis | Main Variables | Planned Method | Proposed Visualization |
|---|---|---|---|---|---|
| Market Activity | How does transaction activity change over time? | Month | Transaction date, count, total amount | Monthly aggregation | Line-chart small multiples |
| Product Performance | Which categories contribute most transaction activity and value? | Product category | Category, count, total amount | Grouped aggregation and ranking | Horizontal-bar small multiples |
| Transaction Distribution | How are individual transaction amounts distributed? | Transaction | Amount | Distribution analysis | Histogram |
| Category Participation | Is recorded gender associated with category participation? | Customer | Gender, purchased/not purchased by category | Customer-level binary participation, Chi-square, Bonferroni correction, Cramér's V | Dumbbell chart plus statistical summary |
| Market Sentiment | How does social-media sentiment composition change over time? | Month | Interaction date, sentiment group | Monthly proportional composition | 100% stacked bar chart |
| Cross-Channel Behavior | Does social engagement correspond with purchasing behavior? | Customer | Interaction count, transaction count, total spend, average amount | Customer aggregation, aligned observation window, correlation analysis | Correlation dot plot plus one supporting relationship plot |

---

# 6. Data Loading and Verification

## 6.1 Required Datasets

Use the finalized cleaned datasets from the preprocessing stage:

- Customer Demographics
- Customer Transactions
- Social Media Interactions

## 6.2 Lightweight Verification

Before visualization, confirm:

- file loaded successfully;
- expected columns are present;
- expected analytical unit is understood;
- missing values have not unexpectedly reappeared;
- exact duplicate rows have not unexpectedly reappeared;
- identifiers expected to be unique remain valid;
- date variables are usable as dates; and
- numerical variables are usable as numeric values.

### Team Output Placeholder

**Verification summary:**  
[Insert verified dataset dimensions and checks here.]

**Any issue found:**  
[Document issue before treatment.]

---

# 7. Analytical Data Preparation

This section should prepare only the derived fields required for Milestone 2.

## 7.1 Date Preparation

Prepare:

- transaction month;
- social-media interaction month; and
- common observation period for transaction-social comparison.

## 7.2 Customer-Level Transaction Summary

Prepare customer-level measures such as:

- transaction count;
- total spending; and
- average transaction amount.

## 7.3 Customer-Level Social Summary

Prepare:

- interaction count per customer.

## 7.4 Product-Category Participation

For each customer and product category:

- **Purchased:** customer purchased the category at least once.
- **Not Purchased:** no recorded purchase in that category.

## 7.5 Sentiment Grouping

Define the approved sentiment grouping to be used in the final temporal composition chart.

### Validation Requirement

After every aggregation or merge, verify:

- analytical unit;
- row count;
- unique customer count where applicable;
- duplicate customer rows where one row per customer is expected; and
- whether the resulting population differs from the source dataset.

---

# 8. Market Activity Over Time

## Analytical Question

**How does transaction activity change over time?**

## Planned Analysis

- Aggregate transactions by month.
- Calculate monthly transaction count.
- Calculate monthly total transaction value.
- Preserve chronological order.
- Avoid claiming seasonality unless the available period supports it.

## Proposed Visualization

**Line-chart small multiples**

- Chart A: monthly transaction count
- Chart B: monthly total transaction value

## Team Tasks

- [ ] Prepare monthly aggregation.
- [ ] Validate month coverage.
- [ ] Create the two charts.
- [ ] Identify meaningful peaks, lows, or changes.
- [ ] Explain whether the pattern represents fluctuation, trend, or insufficient evidence for either.
- [ ] State the FinMark implication without causal claims.

### Result Placeholder

[Insert numerical/visual result.]

### Interpretation Placeholder

[Insert evidence-based interpretation.]

### Business Implication Placeholder

[Insert supported FinMark implication.]

---

# 9. Product-Category Performance

## Analytical Question

**Which product categories contribute most to transaction activity and recorded transaction value?**

## Planned Analysis

For each product category, calculate:

- transaction count;
- total transaction value;
- optional supporting average or median transaction amount.

## Proposed Visualization

**Horizontal-bar small multiples**

- Chart A: transaction count by category
- Chart B: total transaction value by category

## Team Tasks

- [ ] Create category-level summary.
- [ ] Rank categories consistently.
- [ ] Create comparison charts.
- [ ] Determine whether differences are large or modest.
- [ ] Avoid describing the highest category as dominant unless evidence supports that wording.
- [ ] Connect the result to product activity and planning.

### Result Placeholder

[Insert category-level results.]

### Interpretation Placeholder

[Insert evidence-based interpretation.]

### Business Implication Placeholder

[Insert supported FinMark implication.]

---

# 10. Transaction-Amount Distribution

## Analytical Question

**What does the distribution of individual transaction amounts reveal about purchasing behavior?**

## Planned Analysis

Review:

- shape of the distribution;
- center;
- spread;
- zero-value transactions if present;
- tails; and
- possible unusual observations.

## Proposed Visualization

**Histogram**

Optional supporting descriptive statistics may be presented beside the chart.

## Team Tasks

- [ ] Confirm the amount field is numeric.
- [ ] Select a defensible bin count.
- [ ] Create the histogram.
- [ ] Compare visual shape with descriptive statistics.
- [ ] Avoid calling observations outliers unless supported by a defined rule.
- [ ] Explain what the spread means for FinMark.

### Result Placeholder

[Insert distribution result.]

### Interpretation Placeholder

[Insert evidence-based interpretation.]

### Business Implication Placeholder

[Insert supported FinMark implication.]

---

# 11. Gender and Product-Category Participation

## Analytical Question

**Is recorded gender significantly associated with whether a customer purchases from particular product categories?**

Gender is used as a statistical diagnostic rather than as an automatic marketing-targeting variable.

## Unit of Analysis

One customer per category-level participation test.

## Planned Method

For each product category:

1. classify each matched customer as Purchased or Not Purchased;
2. compare participation across recorded gender groups;
3. perform a Chi-square test of independence;
4. apply a Bonferroni correction because multiple category tests are performed;
5. calculate Cramér's V as an effect-size measure; and
6. interpret statistical significance separately from practical importance.

## Hypotheses

**H0:** Product-category participation is independent of recorded gender.

**H1:** Product-category participation is associated with recorded gender.

## Proposed Visualization

**Dumbbell chart**

Display customer participation rates by recorded gender for each product category.

## Team Tasks

- [ ] Build customer-level participation data.
- [ ] Verify one customer contributes once per category.
- [ ] Calculate participation rates.
- [ ] Run the planned statistical tests.
- [ ] Apply the multiple-testing correction.
- [ ] Calculate effect size.
- [ ] Create the dumbbell chart.
- [ ] Distinguish statistical significance from practical importance.
- [ ] Avoid causal or discriminatory targeting claims.

### Statistical Result Placeholder

[Insert Chi-square, adjusted significance, and Cramér's V results.]

### Visualization Result Placeholder

[Insert chart.]

### Interpretation Placeholder

[Insert evidence-based interpretation.]

### Business Implication Placeholder

[Insert supported FinMark implication.]

---

# 12. Social-Media Sentiment Over Time

## Analytical Question

**How does the composition of recorded social-media sentiment change over time?**

## Planned Analysis

- Group interactions by month.
- Calculate the proportion of each approved sentiment group per month.
- Compare composition rather than relying only on raw monthly counts.

## Proposed Visualization

**100% stacked bar chart**

## Team Tasks

- [ ] Confirm sentiment grouping.
- [ ] Calculate monthly sentiment proportions.
- [ ] Verify each month totals 100% after rounding tolerance.
- [ ] Create the composition chart.
- [ ] Identify meaningful changes without overstating normal fluctuations.
- [ ] State any limitation caused by the observation period.

### Result Placeholder

[Insert monthly composition result.]

### Interpretation Placeholder

[Insert evidence-based interpretation.]

### Business Implication Placeholder

[Insert supported FinMark implication.]

---

# 13. Social Engagement and Purchasing Behavior

## Analytical Question

**Does recorded social-media engagement correspond with purchasing behavior?**

## Required Population Rule

Transaction and social-media measures must be compared only after:

1. aligning both datasets to their shared observation period;
2. aggregating each source to one row per customer; and
3. joining only matched customers.

## Planned Measures

Compare interaction count with:

- transaction count;
- total spending; and
- average transaction amount.

## Planned Method

- customer-level aggregation;
- matched-customer merge;
- correlation analysis;
- visual inspection of one representative relationship.

## Proposed Visualization

1. **Correlation dot plot** summarizing the strength and direction of the tested relationships.
2. **One supporting scatter plot** if it adds useful context.

## Team Tasks

- [ ] Define the common observation period from the data.
- [ ] Aggregate transaction activity per customer.
- [ ] Aggregate social activity per customer.
- [ ] Validate the matched population.
- [ ] Calculate the planned correlations.
- [ ] Create the correlation summary.
- [ ] Select at most one supporting scatter plot.
- [ ] Explain the strength of the relationship rather than focusing only on direction.
- [ ] State that correlation does not establish causation.

### Result Placeholder

[Insert correlation results.]

### Interpretation Placeholder

[Insert evidence-based interpretation.]

### Business Implication Placeholder

[Insert supported FinMark implication.]

---

# 14. Statistical and Analytical Validation Summary

Prepare a concise table after all analyses are complete.

| Analytical Question | Method | Population / Unit | Key Statistic or Evidence | Interpretation Status |
|---|---|---|---|---|
| Transaction trend | Monthly aggregation | Month | [To be completed] | [To be completed] |
| Product performance | Grouped aggregation | Product category | [To be completed] | [To be completed] |
| Transaction distribution | Descriptive distribution analysis | Transaction | [To be completed] | [To be completed] |
| Gender × category participation | Chi-square + Bonferroni + Cramér's V | Customer | [To be completed] | [To be completed] |
| Sentiment over time | Monthly proportional composition | Month | [To be completed] | [To be completed] |
| Engagement × purchasing | Correlation analysis | Matched customer | [To be completed] | [To be completed] |

---

# 15. Key Findings

Do not write findings before the analyses are executed and validated.

Use this format for each final finding:

## Finding 1: [Title]

**Finding:**  
[What was observed?]

**Evidence:**  
[What numerical or visual evidence supports it?]

**Interpretation:**  
[What does the evidence mean?]

**Business Implication:**  
[Why does this matter to FinMark?]

Repeat only for findings that are supported strongly enough to deserve inclusion.

---

# 16. Evidence-Based Action Plan

Recommendations should come only after the findings are validated.

Suggested structure:

| Evidence-Supported Finding | Possible FinMark Action | Why the Action Is Supported | Limitation / Caution | Success Measure |
|---|---|---|---|---|
| [Finding] | [Action] | [Evidence link] | [Constraint] | [Measure] |

Do not recommend an action merely because a variable appears in the dataset.

---

# 17. Limitations

The final notebook should evaluate applicable limitations, including:

- incomplete customer overlap across datasets;
- different observation periods;
- matched-subset interpretation for cross-dataset analysis;
- customer-level aggregation simplifying event-level variation;
- correlations measuring association rather than causation;
- limited temporal coverage for seasonality claims;
- high-cardinality or unstandardized location data;
- data-quality issues inherited from the supplied datasets; and
- limits on generalizing findings beyond the FinMark data.

Add or remove limitations based on the final executed analysis.

---

# 18. Future Machine Learning Considerations

This section should not train a model.

Discuss only evidence-supported considerations such as:

- possible target variables;
- candidate explanatory variables;
- variables that showed weak standalone relationships;
- categorical encoding requirements;
- scaling requirements where applicable;
- class imbalance considerations;
- high-cardinality variables;
- matched-population constraints; and
- additional data that may be needed before predictive modeling.

---

# 19. Visualization Selection References

Document the chart-selection references used to justify visualization choices.

Planned references:

- Datylon: Types of Charts and Graphs for Data Visualization
- Deepnote: Types of Graphs
- Flourish: Different Types of Charts
- ThoughtSpot: Types of Charts and Graphs

For the final submission, provide the required citation or reference format specified by the course.

---

# 20. Conclusion

The conclusion should answer:

1. What market trends were identified?
2. What customer-behavior relationships were supported?
3. Which investigated relationships were too weak to drive action?
4. What limitations affect interpretation?
5. How do the visualizations help address the FinMark problem?
6. What should be carried forward into the Terminal Assessment or future machine learning work?

Do not introduce new evidence in the conclusion.

---

# 21. Suggested Team Contribution Plan

This section can be used to divide work without changing the analytical design.

| Work Package | Main Responsibility | Expected Deliverable | Assigned Member |
|---|---|---|---|
| **A. Market Activity** | Sections 8 and 9 | Monthly transaction visuals and product-category comparison | [Assign] |
| **B. Transaction Behavior** | Section 10 | Transaction distribution analysis and interpretation | [Assign] |
| **C. Customer Participation** | Section 11 | Gender-category participation analysis, statistics, and chart | [Assign] |
| **D. Social and Cross-Channel Behavior** | Sections 12 and 13 | Sentiment trend and engagement-purchasing analysis | [Assign] |
| **E. Integration and Quality Review** | Sections 6, 7, 14 to 20 | Verification, consistency, limitations, synthesis, final review | [Assign] |

### Minimum Contribution Standard for Each Assigned Section

Each contributor should provide:

1. the analytical question;
2. the data and unit of analysis used;
3. the data-preparation steps required;
4. the justification for the selected visualization;
5. the completed visualization;
6. the supporting numerical/statistical evidence;
7. the interpretation;
8. the FinMark business implication;
9. the limitation or caution; and
10. confirmation that the section runs correctly in the final notebook.

---

# 22. Final Team Validation Checklist

Before submission, verify:

- [ ] The notebook still addresses the original FinMark problem.
- [ ] Milestone 2 extends Milestone 1 rather than contradicting it.
- [ ] Only finalized cleaned datasets are used.
- [ ] Every visualization answers a defined analytical question.
- [ ] Units of analysis are correct.
- [ ] Cross-dataset analyses use appropriate aggregation and matched populations.
- [ ] Transaction-social comparisons use aligned observation periods.
- [ ] Statistical tests are used only where justified.
- [ ] Statistical significance is distinguished from effect size or practical importance.
- [ ] No correlation is described as causation.
- [ ] Charts follow the FinMark Visualization Standard.
- [ ] Findings are supported by visible evidence.
- [ ] Recommendations follow from validated findings.
- [ ] Limitations are stated.
- [ ] Team contributions are identifiable.
- [ ] The final notebook runs from top to bottom in a fresh Colab runtime.
- [ ] No placeholder text remains in the submission version.

---

# Proposed Final Analytical Flow

```text
FINMARK BUSINESS PROBLEM
        ↓
MILESTONE 1 EDA FOUNDATION
        ↓
WEEK 7 VISUALIZATION EXPLORATION
        ↓
FINAL CLEANED DATA VERIFICATION
        ↓
ANALYTICAL DATA PREPARATION
        ↓
MARKET ACTIVITY VISUALIZATION
        ↓
PRODUCT PERFORMANCE VISUALIZATION
        ↓
TRANSACTION DISTRIBUTION
        ↓
CUSTOMER PARTICIPATION ANALYSIS
        ↓
SOCIAL SENTIMENT ANALYSIS
        ↓
CROSS-CHANNEL BEHAVIOR ANALYSIS
        ↓
STATISTICAL / ANALYTICAL VALIDATION
        ↓
FINDINGS
        ↓
INTERPRETATION
        ↓
BUSINESS IMPLICATIONS
        ↓
LIMITATIONS
        ↓
FUTURE ML CONSIDERATIONS
        ↓
CONCLUSION
```
