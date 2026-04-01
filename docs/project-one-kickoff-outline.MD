Overview

This is the first step in building DemandCast — your individual machine learning project that predicts hourly taxi demand by pickup zone. Before any modeling can happen, you need a clean, reproducible project environment and a working understanding of the dataset you'll be building on for the next four weeks.

In this assignment, you will set up your project repository, organize it for professional ML development, and conduct an initial exploration of the NYC taxi dataset. The decisions and observations you make here will directly inform your feature engineering and modeling work in the weeks ahead.

You are working in a pair for Project 1 — both you and your partner will complete your own standalone versions of this assignment. You are encouraged to troubleshoot setup issues together and share observations about the dataset, but each student submits their own repository with their own work.

Instructions

Part 1 — Project Repository Setup

Create a GitHub repository named aml-demandcast and set it up according to the Project 1 Repository Structure Guide (see attached). This includes:

The required folder structure
A .gitignore configured correctly for an ML project (data files, model artifacts, virtual environment, and notebook checkpoints should not be tracked)
A requirements.txt listing the project dependencies
A README.md with a first draft describing what the project does, what data it uses, and what it predicts
Once the structure is in place, create and activate a virtual environment, install your dependencies, and make your first commit with the message: scaffold: initial project structure.

Part 2 — Dataset Exploration

Download the January 2025 NYC Yellow Taxi trip data (parquet format) from the NYC TLC Trip Record Data page and place it in your data/ folder. This folder is gitignored — the data file itself will not be committed.

Open a notebook at notebooks/01_initial_exploration.ipynb and use it to explore the dataset. Your goal is not to produce a polished analysis — it is to get familiar with the data before Week 2's structured EDA. At minimum, your notebook should address:

The shape of the dataset and what each column represents
What data quality issues you would expect to find, and how you might detect them
Which columns are most likely to be useful for predicting hourly taxi demand by pickup zone, and why
Use your AI coding assistant to help generate code where useful — refer to the Using AI Assistance in This Course guide for expectations around how to use and document it.

Commit the notebook to your repository before the deadline.

Learning Outcomes

This assignment addresses the following course learning outcomes:

Apply supervised learning algorithms, feature engineering techniques, and model evaluation methods to real-world datasets — by identifying which raw features are relevant to the prediction task and why.
Build and evaluate end-to-end machine learning pipelines that transform raw data into deployed predictions — by establishing the project foundation that the full pipeline will be built on.
Deliverables

Submit the link to your aml-demandcast GitHub repository. At the time of submission, the repository must contain:

A complete folder structure as specified in the Project 1 Repository Structure Guide
A .gitignore, requirements.txt, and README.md committed
notebooks/01_initial_exploration.ipynb committed with dataset observations and AI-assisted exploration code
Referenced Documents

Project 1 Repository Structure Guide — folder structure, .gitignore contents, and requirements.txt starting point