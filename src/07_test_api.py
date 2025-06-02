import requests
import pandas as pd
import json

# Prepare the data
data = pd.read_csv("data/fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')

# Convert data to JSON format
json_data = {
    "dataframe_split": {
        "columns": X.columns.tolist(),
        "data": X.head(2).values.tolist()  # Testing with 2 rows
    }
}

# Send request to the API
response = requests.post(
    url="http://localhost:5002/invocations",
    json=json_data,
    headers={"Content-Type": "application/json"}
)

# Display predictions
if response.status_code == 200:
    predictions = response.json()
    print("\nReceived predictions:")
    print(predictions)
else:
    print(f"Error: {response.status_code}")
    print(response.text)