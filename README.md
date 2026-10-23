# SignalMind

SignalMind is a business analytics application built with Python and Streamlit. It analyzes structured business data and presents report-based insights.

## Features

- **Sales Report:** Analyze sales trends and order activity.
- **Revenue Report:** Reconcile recorded payments against item sales and freight, highlighting differences for investigation.
- **Product Performance Report:** Compare product and category performance.
- **Business Performance Report:** Combine indicators from the sales, revenue, and product analyses.
- **Gita Intelligence:** Integrates machine-learning functionality for supported anomaly-detection and analysis workflows.
- **Dataset Upload:** Upload supported CSV and Excel files for data validation and analysis.

## Technology

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- joblib

## Run locally

1. Install Python.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   python -m streamlit run main.py
   ```

## Data requirements

The built-in Olist reports read CSV files from the project's `data` directory. The configured reports require the relevant source files to be present. Uploaded-dataset workflows may have separate requirements.

## Important limitations

Revenue differences are investigation flags, not proof of financial loss. Machine-learning anomaly scores should not be interpreted as confirmed errors or fraud.

## Project status

Developed as a business analytics and machine-learning demonstration project.
