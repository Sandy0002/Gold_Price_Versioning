import evidently
import pandas as pd
# from src_versioning.data_preprocess import fetch_data_postgres
from data_preprocess import fetch_data_postgres,get_engine,inspect
import os
import json
import datetime
from sqlalchemy import inspect, text

from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import datetime



'''Storing drifted features
Columns like below:
date | drift_detected | drift_score | features_drifted
'''


def store_drift_data(n_features, drifted_features):
#     table_name = os.getenv("DRIFT_TABLE", "drift_monitoring")
    table_name = os.getenv("DRIFT_TABLE")
    engine = get_engine()

    inspector = inspect(engine)
    table_exists = table_name in inspector.get_table_names()

    # Convert drifted features list → JSON
    drift_json = json.dumps(drifted_features)

    # Create a DataFrame representing one row (used if table doesn't exist)
    df = pd.DataFrame([{
        "timestamp": datetime.datetime.today(),
        "num_features_drifted": n_features,
        "features_drifted": drift_json
    }])

    if not table_exists:
        # Create table + insert first row
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"✅ Table '{table_name}' created and first drift row inserted.")

    else:
        # Append a new drift record
        df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"✅ Drift record appended → '{table_name}'.")



def detector():
    reference_df = pd.read_parquet("data/reporting_data/base_data.parquet")
    current_df = fetch_data_postgres()
    table_name = os.getenv("TABLE_NAME")

    # =========================================================
    # 1️⃣ GENERATE FULL DRIFT REPORT (for visualization)
    # =========================================================
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    drift_report.save_html("data/reporting_data/drift_full_report.html")

    drift_report_dict = drift_report.as_dict()
    dataset_drift_info = drift_report_dict["metrics"][0]["result"]

    dataset_drifted = dataset_drift_info["dataset_drift"]


    # =========================================================
    # 2️⃣ RUN TEST SUITE FOR PASS/FAIL SIGNAL
    # =========================================================
    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(reference_data=reference_df, current_data=current_df)

    suite.save_html("data/reporting_data/drift_test_suite_report.html")
    suite_results = suite.as_dict()

    suite_pass = suite_results["summary"]["all_passed"]
    drift_alert = not suite_pass  # True if drift detected by test suite


    # =========================================================
    # 3️⃣ FINAL DRIFT DECISION + STORAGE
    # =========================================================
    if drift_alert:
      num_drifted = dataset_drift_info["number_of_drifted_columns"]
      drifted_features = dataset_drift_info["drifted_features"]

      print("⚠️ Drift detected by Test Suite!")
      print(f"Drifted features count: {num_drifted}")
      print(f"Drifted features: {drifted_features}")

        # store in your DB using your existing function
      store_drift_data(
            drifted_features_count=num_drifted,
            drifted_features=drifted_features
        )
    else:
      print("✅ No drift detected by Test Suite.")



if __name__=='__main__':
      detector()