# DemandCast Phase 2 Notebook Brief
## File Target
Implement work in:

`notebooks/01_initial_exploration.ipynb`

This notebook is for the **initial exploration** assignment only. It is not meant to be a full polished EDA, a feature engineering pipeline, or a modeling notebook.

---

## Assignment Context
The project is **DemandCast**, a machine learning project that will predict **hourly taxi demand by pickup zone**.

The current assignment asks for:
- an initial exploration of the January 2025 NYC Yellow Taxi dataset
- a working understanding of the dataset structure
- identification of likely data quality issues
- discussion of which columns are useful for predicting hourly taxi demand by pickup zone

This notebook should help establish the foundation for later feature engineering and modeling work.

---

## Core Framing
This notebook must stay aligned to the actual prediction task:

**Predict hourly taxi demand by pickup zone**

That means the notebook should not become generic trip-level exploration for its own sake.

A key point to make explicit in the notebook:
- the raw taxi dataset is trip-level data
- the likely future modeling dataset will be aggregated to something like:
  - `pickup_hour`
  - `PULocationID`
  - `trip_count`
- the purpose of this notebook is to understand the raw data well enough to support that later transformation

---

## Scope
The notebook should do the following:

1. Load the January 2025 NYC Yellow Taxi parquet file from the `data/` folder
2. Show the dataset shape, columns, dtypes, and a small sample of rows
3. Briefly explain what the main columns represent
4. Perform a simple, useful data quality audit
5. Explore distributions and time-based demand patterns relevant to forecasting
6. Create a preview aggregated demand table at the zone-hour level
7. Explain which fields are likely useful for prediction and which are likely not useful or may cause leakage
8. End with concise findings and next steps

The notebook should **not**:
- build a model
- perform heavy feature engineering
- create a production pipeline
- add unnecessary abstractions
- create extra Python modules unless absolutely necessary
- turn into an overly long or overdesigned report

---

## Required Notebook Structure
Use clear markdown section headers. Keep the notebook readable and minimal.

Recommended structure:

1. **Title and Objective**
2. **Load Data**
3. **Dataset Shape, Columns, and First Look**
4. **Column Meaning and Initial Interpretation**
5. **Data Quality Checks**
6. **Basic Descriptive Summaries**
7. **Time-Based Demand Exploration**
8. **Pickup Zone Demand Aggregation Preview**
9. **Likely Useful Features for Demand Prediction**
10. **Key Findings and Next Steps**

---

## Required Analysis Content

### 1. Load and Inspect the Dataset
Include:
- parquet load from `data/`
- dataset shape
- column list
- dtypes
- `head()`

Keep this straightforward.

---

### 2. Column Meaning
Add a markdown section that explains the major fields in plain English.

Focus especially on:
- pickup datetime
- dropoff datetime
- pickup location ID
- dropoff location ID
- passenger count
- trip distance
- fare/payment-related columns
- flags/categorical fields that may matter

Do not overdo this. Keep it readable.

---

### 3. Data Quality Checks
At minimum include:

- null counts by column
- null percentages by column
- duplicate row count
- check for pickup datetime after dropoff datetime
- check for zero or negative trip duration
- create trip duration in minutes for inspection
- check for zero or negative trip distance
- inspect suspicious passenger counts
- inspect suspicious negative fare/total-related values where relevant
- inspect missing or invalid pickup zone values if present

This section should explain what issues would matter for later modeling and why.

---

### 4. Basic Summaries
Include simple useful summaries such as:
- numeric `.describe()`
- selected categorical value counts where useful
- brief interpretation in markdown

Do not dump excessive output without explanation.

---

### 5. Time-Based Demand Exploration
This is a core section.

Include plots or summaries for:
- trips by hour of day
- trips by day of week
- trips over time across the month if practical
- top pickup zones by trip count

Also include one demand-focused heatmap:
- hour of day vs day of week using trip counts

The point is to identify temporal demand patterns relevant to forecasting.

---

### 6. Pickup Zone Demand Aggregation Preview
Create a small aggregated dataset preview at the likely modeling grain:

- floor pickup timestamp to hour
- group by `pickup_hour` and `PULocationID`
- count trips as something like `trip_count`

Then inspect:
- shape of the aggregated table
- first few rows
- summary of `trip_count`
- highest-demand pickup zones
- highest-demand hours

This section is important because it directly connects raw trip data to the eventual ML task.

---

### 7. Feature Usefulness Discussion
Add a markdown section explaining which columns are likely useful for predicting hourly demand by pickup zone.

Should identify as likely useful:
- pickup datetime-derived features
  - hour of day
  - day of week
  - weekend / weekday
- `PULocationID`
- later lagged demand features derived from aggregation

Should explain that many trip-level columns are weaker or not appropriate as direct predictors for future demand, such as:
- trip distance
- fare amount
- tip amount
- total amount
- payment type
- dropoff-related outcome fields

The notebook should explicitly mention the difference between:
- variables available before the prediction window
- variables that are outcomes of completed trips
- variables that may create leakage

---

### 8. Key Findings
End with a concise markdown summary covering:
- biggest data quality concerns noticed so far
- strongest early signals for demand prediction
- the likely unit of analysis for modeling
- what should be explored next in later EDA / feature engineering

---

## Coding Standards
Keep the notebook implementation simple and high quality.

### General rules
- prefer clear, direct pandas code
- avoid unnecessary helper classes or abstraction layers
- avoid overengineering
- avoid excessive functions unless they clearly reduce repetition
- use short, readable cells
- keep variable names clear and conventional
- include brief comments only where useful
- do not build a custom framework for EDA

### Notebook style
- use markdown to explain purpose and findings
- do not produce giant noisy output dumps
- do not include dozens of redundant plots
- do not include advanced visual styling unless it is simple and improves readability
- keep plots course-appropriate and minimal

### Plotting
Use a small number of useful plots only.

Recommended:
- histograms for trip duration / trip distance
- bar charts for pickup demand by hour, day of week, top zones
- one heatmap for hour vs day-of-week demand

Do not include:
- large pairplots
- bloated correlation analysis as the main focus
- plots unrelated to the forecasting target

---

## Implementation Constraints
- Work only in the notebook unless a very small supporting change is clearly necessary
- Do not change the repository structure
- Do not rewrite unrelated project files
- Do not add unnecessary dependencies
- Do not assume a specific parquet filename if it can be discovered from `data/`
- Keep the notebook suitable for a machine learning class assignment, not a production analytics report

---

## Expected Outcome
When this is done, the notebook should:
- satisfy the assignment requirements
- show a clear understanding of the dataset
- stay tightly connected to the hourly demand prediction problem
- remain simple, readable, and easy to present or revise later