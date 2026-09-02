# Stock Movement Prediction Platform

Project ini merupakan platform prediksi pergerakan saham dan backtesting.

## Track A (Data Pipeline, Database, & Baseline ML)
- **Data Collection**: OHLCV raw data scraping (`src/data/collector.py`)
- **Data Cleaning**: Time-series cleaning & validation (`src/data/cleaner.py`)
- **EDA**: Exploratory data analysis (`src/data/eda.py`)
- **Database**: PostgreSQL schema & migration (`src/database/`)
- **Data Preparation**: Target variable ($R_{t+1}$) & strict chronological time-series split (`src/data/preparation.py`)

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
