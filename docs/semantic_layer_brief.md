# DemandCast Semantic Layer Brief
## Target File
Create the semantic layer at:

`semantic/semantic-layer.yaml`

Create the `semantic/` folder if it does not already exist.

---

## Purpose
This semantic layer should give the project a clean, reusable understanding of the NYC Yellow Taxi dataset for the DemandCast machine learning project.

The project goal is to predict:

**hourly taxi demand by pickup zone**

The semantic layer is not meant to be a full enterprise warehouse spec. It should be a simple, readable YAML file that helps future work stay consistent across:
- exploration
- feature engineering
- modeling
- evaluation

It should reflect:
- the raw yellow taxi trip dataset
- the forecasting target
- the likely modeling grain
- which fields are useful, less useful, or leakage-prone

---

## Source Documents
Use these as source references while building the semantic layer:

- `docs/data_dictionary_trip_records_yellow.pdf`
- `docs/project-one-kickoff-outline.MD`
- `docs/01_initial_exploration_notebook_brief.md`

Also inspect the actual data file if needed to verify column names and types:
- `data/yellow_tripdata_2025-01.parquet`

The data dictionary should be treated as the main source of truth for field meanings.
The parquet file should be used to confirm actual available columns and practical dtypes.
The project outline and notebook brief should be used to keep the semantic layer aligned to the ML objective.

---

## Core Framing
The semantic layer must stay aligned to the project’s prediction task:

**predict hourly taxi demand by pickup zone**

That means the semantic layer should make clear:
- the raw data is at the **trip level**
- the likely modeling dataset is at the **zone-hour level**
- the core target is a trip-count style measure such as hourly demand by pickup zone

A key modeling concept should be represented clearly:

- raw grain: one row per taxi trip
- modeling grain: one row per `pickup_hour` + `PULocationID`
- target: `trip_count`

---

## Scope
The semantic layer should define:

1. dataset-level metadata
2. source file(s)
3. raw grain
4. modeling grain
5. entities
6. time dimensions
7. key raw dimensions
8. key measures
9. derived features / semantic concepts
10. target variable
11. exclusions / leakage-prone fields
12. assumptions / notes

Keep this practical and compact.

---

## Design Requirements

### 1. Keep it simple
This file should be easy for a human or coding assistant to read.

Do not:
- create a giant enterprise ontology
- overengineer the YAML
- include dozens of speculative fields that are not needed
- invent complex metric frameworks

Do:
- keep the structure clean
- use descriptive names
- use exact raw column names where possible
- separate raw fields from derived modeling concepts

---

### 2. Reflect the actual ML problem
The semantic layer should help answer:
- What is the raw unit of data?
- What is the future modeling unit?
- What is the target?
- Which fields are likely useful?
- Which fields are likely leakage or post-outcome variables?

This project is about demand forecasting, not trip outcome prediction.

---

### 3. Use exact raw field names
For raw columns:
- use the exact field names found in the parquet file / data dictionary
- do not rename raw columns arbitrarily inside the semantic layer
- if you include a friendlier label, keep the raw column name explicit

---

### 4. Include derived semantic concepts
The semantic layer should include a section for derived modeling fields such as:
- `pickup_hour`
- `pickup_date`
- `pickup_day_of_week`
- `pickup_hour_of_day`
- `is_weekend`
- `trip_duration_min`
- `trip_count`

These should be marked as derived rather than raw.

---

### 5. Make leakage explicit
The semantic layer should clearly note that some trip-level columns are not appropriate as direct predictors for future hourly demand.

Include a section that marks fields like these as likely leakage-prone or low-value for direct forecasting use:
- dropoff-related fields
- fare outcomes
- tip amounts
- payment outcomes
- total charged amounts
- other trip-completion outcome variables

Do not say they are useless in all contexts. Say they are weaker or inappropriate as direct predictors for **future demand forecasting**.

---

## Recommended YAML Structure
Use a structure similar to this conceptually, but keep it clean and readable:

- project
- dataset
- sources
- raw_grain
- modeling_grain
- target
- entities
- time_dimensions
- dimensions
- measures
- derived_features
- predictive_feature_groups
- leakage_or_post_outcome_fields
- notes

The exact formatting can vary, but it should remain minimal and logically organized.

---

## What to capture

### Project metadata
Include:
- project name: DemandCast
- objective: predict hourly taxi demand by pickup zone
- dataset name
- source parquet path
- reference docs used

---

### Raw grain
State clearly:
- one row represents one taxi trip record

---

### Modeling grain
State clearly:
- one row in the future modeling table is expected to represent one pickup zone in one hour

Suggested semantic keys:
- `pickup_hour`
- `PULocationID`

---

### Target
Define a target concept like:
- `trip_count`
- description: number of taxi pickups in a given pickup zone during a given hour
- derived from grouping trip records by `pickup_hour` and `PULocationID`

---

### Entities and dimensions
Capture the most relevant dimensions such as:
- pickup location
- dropoff location
- vendor
- payment type if present
- passenger count
- flags/categorical operational fields

Keep the focus on dimensions that help interpret the dataset or may matter later.

---

### Time dimensions
Include:
- pickup timestamp
- dropoff timestamp
- pickup_hour
- pickup_date
- pickup_day_of_week
- pickup_hour_of_day
- weekend / weekday concept

Time is central to this project and should be modeled clearly.

---

### Measures
Include measures that are meaningful for exploration and modeling setup, such as:
- trip_count
- trip_distance
- trip_duration_min
- fare_amount
- tip_amount
- total_amount

Where relevant, note whether the measure is:
- raw
- derived
- likely useful for forecasting
- more useful for QA / descriptive analysis than direct prediction

---

### Predictive feature groups
Include a clean section grouping likely feature usefulness into categories such as:
- strong candidate predictors
- contextual / secondary predictors
- likely leakage or post-outcome fields

This section should align with the ML objective.

---

### Notes
Add short notes for things like:
- this semantic layer is for project-level consistency, not warehouse governance
- raw trip data must be aggregated to zone-hour for the forecasting task
- some features may be used later as lagged aggregates even if they are not useful as direct raw predictors

---

## Quality Standards
- simple YAML
- readable indentation
- minimal but useful descriptions
- no unnecessary nesting
- no speculative complexity
- no invented columns not supported by the data dictionary or parquet file

If a field is uncertain:
- verify it against the parquet file or data dictionary
- if still uncertain, omit it rather than inventing it

---

## Expected Outcome
The final `semantic/semantic-layer.yaml` should:
- help future notebook and feature engineering work stay consistent
- reflect the real forecasting target
- distinguish raw fields from derived modeling concepts
- identify likely useful predictors vs leakage-prone fields
- remain simple enough for a student ML repo