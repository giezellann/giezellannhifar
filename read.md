# Campus Facility Reservation Monitoring — Streamlit

A Streamlit dashboard based on the uploaded Jupyter Notebook for campus facility reservation data.

## Files
- `pp.py` — Streamlit application
- `requiremnts.txts` — Python package list (filename kept as requested)
- `read.md` — this guide

## Run on your computer
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:
   ```bash
   pip install -r requiremnts.txts
   ```
4. Start the app:
   ```bash
   streamlit run pp.py
   ```
5. Open the local URL printed in the terminal.

## Run on Streamlit Community Cloud
Streamlit Cloud normally expects a file named `requirements.txt`. Since you requested `requiremnts.txts`, rename that file to `requirements.txt` before deploying. Set the app's main file path to `pp.py`.

## Dataset
Upload a CSV from the sidebar. The expected columns are:
- `Facility_Type`
- `User_Category`
- `Lead_Time_Days`
- `Participants`
- `Usage_Status`
- `Actual_Attendance`

Optional columns such as `Reservation_ID` and `Cancellation_Days_Before` are retained if present. If no CSV is uploaded, the app can use generated sample data for demonstration.

## Dashboard features
- Dataset overview and summary statistics
- Data cleaning and feature engineering
- Facility and reservation outcome analysis
- Decision Tree classification metrics and feature importance
- Download cleaned CSV data
