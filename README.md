# DemandCast (Project 1)

## What Is This Project?

DemandCast is an individual applied machine learning project focused on predicting hourly taxi demand by NYC pickup zone.

This repository is the foundation for a multi-week end-to-end ML workflow:

- repository setup and reproducible environment
- dataset exploration and feature engineering
- model training, evaluation, and tracking with MLflow
- dashboard delivery with Streamlit

Project 1 begins by building a clean, professional ML repo and understanding the dataset before full modeling begins.

## What Is the Data?

The project uses:

- NYC TLC Yellow Taxi Trip Records
- January 2024
- Parquet format (`yellow_tripdata_2024-01.parquet`)

Data source:

- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Expected local file location:

- `data/yellow_tripdata_2024-01.parquet`

Notes:

- The `data/` folder is intentionally gitignored.
- Raw data files are kept locally and are not committed to GitHub.

## What Are We Predicting?

The prediction target is hourly taxi demand per pickup zone.

In practice, demand is created by aggregating trip records into zone-hour buckets, for example by grouping on:

- `PULocationID`
- pickup datetime rounded/grouped to hourly intervals

This creates a supervised learning target (`demand`) for forecasting future hourly demand.

## Project Structure

```text
aml-demandcast/
├── data/                    # gitignored raw/local data
├── notebooks/
│   ├── 01_initial_exploration.ipynb
│   └── 02_eda_skeleton.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   ├── cv.py
│   └── tune.py
├── models/                  # gitignored model artifacts
├── app/
│   └── dashboard.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Place the January 2024 parquet file in `data/`.
4. Start with `notebooks/01_initial_exploration.ipynb` for initial data understanding.

## Week 1 Focus (This Assignment)

- Scaffold the repository for ML development.
- Configure `.gitignore`, dependencies, and README.
- Perform first-pass dataset exploration in `notebooks/01_initial_exploration.ipynb`.
- Document initial observations that will guide Week 2 feature engineering and EDA decisions.

## Learning Outcomes Alignment

This project setup and exploration supports:

- selecting relevant supervised learning features for a real-world prediction task
- building toward a full end-to-end ML pipeline from raw data to deployable predictions

## Deliverables Snapshot

At submission time, this repository should contain:

- required project folder structure
- committed `.gitignore`, `requirements.txt`, and `README.md`
- committed `notebooks/01_initial_exploration.ipynb` with dataset observations and AI-assisted exploration code

## Notes on Collaboration

Project 1 is completed individually. You may collaborate on troubleshooting and discussion, but each student submits their own standalone repository and work.

