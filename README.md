Romanian Real Estate Market Analysis
This repository contains a practical data analysis and machine learning project applied to the local real estate market. The goal is to analyze residential price trends and identify correlations with macroeconomic indicators, starting from a dataset built from scratch.

To provide real-world context, I extracted real estate listings via web scraping and combined them with official state data: IRCC interest rate trends (from the BNR) and housing price/building permit indices (from the INS).

Analysis Overview
The project logic is divided into dedicated notebooks, each addressing a specific problem:

Exploratory Data Analysis (EDA): I processed the raw data, handled missing values, removed outliers (incorrect listings), and created initial visualizations to examine the distribution of prices per square meter at the county level.

Multiple Regression: I trained a model to estimate/predict apartment prices based on physical characteristics such as surface area, number of rooms, and location.

Time Series Analysis (ARIMA): I analyzed historical price trends and seasonality. This section highlights how IRCC increases influence market volume and pricing.

Clustering (K-Means): An unsupervised analysis used to segment listings and neighborhoods into groups (e.g., Budget, Standard, Premium).

Tools Used
Python: Core language.

Data Manipulation & Analysis: Pandas, NumPy.

Data Collection (Scraping): Custom script located in src/scraper.py using standard scraping libraries.

Machine Learning & Statistics: Scikit-Learn (for Regression and K-Means) and Statsmodels (for Time Series/ARIMA).

Data Visualization: Matplotlib and Seaborn for notebook charts.

Dashboard: Streamlit (a simple web application to interact with the data).

Project Structure
Plaintext
ro-real-estate-analysis/
├── data/
│   ├── raw/                 # raw data (imobiliare_scrape_2024.csv, INS & BNR data)
│   └── processed/           # cleaned, normalized data ready for modeling
├── notebooks/
│   ├── 01_EDA.ipynb         # initial exploration and visualization
│   ├── 02_regresie_multipla.ipynb
│   ├── 03_serii_de_timp.ipynb # ARIMA, trends, seasonality
│   └── 04_clustering_kmeans.ipynb
├── src/
│   ├── scraper.py           # data collection logic
│   ├── etl.py               # cleaning, normalization, and export to data/processed
│   ├── models.py            # functions for regression, ARIMA, and clustering
│   └── dashboard.py         # Streamlit application code
├── outputs/
│   ├── grafice/             # .png files exported from notebooks
│   └── raport_final.pdf     # final report with analysis conclusions
├── requirements.txt
├── .gitignore
└── README.md
