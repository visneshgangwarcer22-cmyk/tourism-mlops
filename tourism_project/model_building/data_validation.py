
import json
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
df = pd.read_csv(DATA_PATH)

required_columns = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
}
missing_columns = sorted(required_columns - set(df.columns))
assert not missing_columns, f"Missing required columns: {missing_columns}"
assert df["ProdTaken"].isin([0, 1]).all(), "Target must contain only 0/1."
assert df["CustomerID"].is_unique, "CustomerID must be unique."

report = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_distribution": df["ProdTaken"].value_counts().sort_index().to_dict()
}
out = Path(__file__).resolve().parent / "data_validation_report.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
