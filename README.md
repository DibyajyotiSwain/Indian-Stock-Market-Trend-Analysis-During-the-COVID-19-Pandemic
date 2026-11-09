
# Indian Stock Market Trend Analysis During the COVID-19 Pandemic

### An Interactive Data Analytics Dashboard for Historical Stock Market Analysis

<p align="center">
DIBYAJYOTI SWAIN | IBCP Career-related Programme | Class XII
</p>

---

## Project Information

| Detail | Information |
|---|---|
| **Project Title** | Indian Stock Market Trend Analysis During the COVID-19 Pandemic |
| **Submitted by** | **Dibyajyoti Swain** |
| **Programme** | IBCP |
| **Class** | XII |
| **Institution** | Sai International School, Bhubaneswar |
| **Project Domain** | Data Analytics, Financial Markets and Machine Learning |
| **Study Period** | January 2020 – November 2021 |

---

## 1. Introduction

The COVID-19 pandemic brought significant uncertainty to global financial markets. Lockdowns, economic disruptions, changing investor sentiment and unprecedented market conditions influenced stock market movements across the world.

This project, **Indian Stock Market Trend Analysis During the COVID-19 Pandemic**, explores historical Indian stock market data to understand market behaviour during this extraordinary period.

The project transforms a large historical dataset into meaningful financial insights through data cleaning, statistical analysis, interactive visualisation and machine learning.

An interactive dashboard has been developed to make complex financial information easier to explore, interpret and compare.

## 2. Problem Statement

Analysing a large volume of historical stock market data can be complex and time-consuming.

Closing prices alone do not provide a complete understanding of market performance or financial risk. Different stocks may respond differently to changing economic conditions, particularly during periods of uncertainty.

The project addresses these challenges by developing a data-driven analytical system that presents historical stock performance, returns, volatility, drawdowns and correlations through an interactive dashboard.

## 3. Project Objectives

- Analyse historical Indian stock market data during the COVID-19 period.
- Clean and organise a large financial dataset using Python.
- Calculate daily returns and moving averages.
- Measure stock volatility and historical risk.
- Compare stock performance and maximum drawdown.
- Examine correlations among selected stocks.
- Identify historical patterns in positive and negative trading days.
- Develop an interactive dashboard for financial data visualisation.
- Explore machine learning for experimental next-day market direction classification.

## 4. Dataset Description

The project uses historical Indian stock market data covering:

- **Study period:** January 2020 to November 2021.
- **Approximate observations:** 698,519, as reported in the project presentation.
- **Cleaned dataset size:** Approximately 32 MB.

### Key dataset fields

| Field | Description |
|---|---|
| `Symbol` | Stock identification symbol |
| `Date` | Trading date |
| `Closing_Price` | Closing price of the stock |
| `Daily_Return` | Daily percentage return |

The dataset undergoes cleaning and preprocessing before being used for statistical analysis and dashboard visualisation.

### Dataset availability

The complete cleaned CSV file is approximately 32 MB and is hosted separately to keep the GitHub repository lightweight.

**Full dataset:** [https://drive.google.com/drive/folders/1pugHkHHPrg3KPThZ2rauUbIBLfl9fkcW]

**PRESENTATION(PPT):** [https://drive.google.com/drive/folders/1pugHkHHPrg3KPThZ2rauUbIBLfl9fkcW]

A smaller sample dataset can be included in this repository for testing and demonstration.

## 5. Methodology

The project follows a structured analytical workflow:

1. **Historical Data:** Collection and examination of historical stock market records.
2. **Data Cleaning:** Handling missing values, organising records and preparing the dataset.
3. **Data Processing:** Calculating daily returns and other analytical metrics.
4. **Statistical Analysis:** Examining performance, volatility, drawdowns and correlations.
5. **Visualisation:** Creating charts and comparative financial visualisations.
6. **Interactive Dashboard:** Integrating analytical results into an interactive web-based dashboard.

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical calculations |
| Plotly | Interactive charts and visualisations |
| Dash | Interactive dashboard development |
| Scikit-learn | Machine learning and model evaluation |
| Jupyter Notebook | Data exploration and development |
| HTML / CSS | Dashboard presentation and styling |

## 7. Key Features

### Market Overview
- Key market performance indicators.
- Summary of historical market behaviour.
- Overview of selected stock data.

### Stock Explorer
- Interactive stock selection.
- Historical closing-price visualisation.
- Configurable historical date ranges.
- Moving-average comparisons.

### Performance Analysis
- Daily return analysis.
- Historical return comparisons.
- Yearly performance analysis.
- Positive versus negative trading-day comparisons.

### Risk Analysis
- Volatility measurement.
- Maximum drawdown analysis.
- Risk and return comparison.
- Historical stock performance assessment.

### Correlation Analysis
- Correlation matrix.
- Comparative analysis of selected stocks.
- Identification of relationships between stock returns.

### Machine Learning
- Random Forest classification model.
- Experimental next-day market direction classification.
- Model evaluation using historical data.

*Machine learning outputs are experimental analytical results, not guaranteed predictions of future stock prices.*

## 8. Dashboard Preview

The dashboard is designed to bring different aspects of stock market analysis into one interactive interface.

It enables users to explore historical prices, compare returns, examine volatility, analyse drawdowns and investigate relationships between selected stocks.

**Note: Screenshots Present In Screenshots Folder**

## 9. Project Structure

```text
Indian-Stock-Market-Analysis/
│
├── app.py
│
├── data/
│   └── README.md
│
├── sample_data/
│   └── stock_data_sample.csv
│
├── assets/
│   └── style.css
│
├── screenshots/
│   ├── dashboard-overview.png
│   ├── stock-analysis.png
│   └── risk-analysis.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

The complete cleaned dataset is stored externally rather than being committed to the standard Git repository.

## 10. Installation and Setup

### Prerequisites

- Python installed on your computer.
- Git installed (if cloning the repository).
- A compatible Python environment.
- The complete cleaned dataset or a compatible sample dataset.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project directory

```bash
cd Indian-Stock-Market-Analysis
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Add the dataset

Download the cleaned dataset using the link provided in the Dataset Availability section.

Place it in the following location:

```text
data/All_Stocks_Data_Cleaned.csv
```

### Step 6: Run the dashboard

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:8050
```

## 11. Expected Outcomes

The project aims to provide a clearer understanding of historical Indian stock market behaviour during the COVID-19 period.

It demonstrates how large financial datasets can be processed and transformed into useful visual insights.

The dashboard brings together historical performance, risk measurement, correlation analysis and experimental machine learning in one platform.

## 12. Limitations

- The analysis is limited to the historical dataset and its available fields.
- Historical market behaviour does not establish that COVID-19 alone caused a particular market movement.
- Machine learning results depend on the quality and scope of historical data.
- Experimental classification results should not be interpreted as reliable investment signals.
- Historical returns do not guarantee future performance.

## 13. Future Scope

Possible improvements include:

- Extending the dataset to include pre-pandemic and post-pandemic periods.
- Adding sector-wise and index-level comparisons.
- Incorporating trading volume and additional financial indicators.
- Improving dashboard interactivity.
- Exploring additional machine learning approaches.
- Integrating regularly updated market data.

## 14. Academic Significance

This project demonstrates the practical application of data analytics, statistical methods and visualisation techniques to a real-world financial research problem.

It also provides practical experience in Python programming, data preprocessing, financial analysis, dashboard development and introductory machine learning.

## 15. Disclaimer

This project is developed for educational and analytical purposes only. It does not provide financial advice, investment recommendations or guaranteed stock market predictions.

---

## Author

**Dibyajyoti Swain**  
IBCP, Class XII  
Sai International School, Bhubaneswar

**Project:** Indian Stock Market Trend Analysis During the COVID-19 Pandemic

---

*Thank you for exploring this project.*
