import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop identifier/index-like columns if present
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True, errors="ignore")

# NOTE: Categorical features like 'TypeofContact', 'Occupation', 'ProductPitched'
# are intentionally left as raw strings. The training pipeline will one-hot-encode them,
# and the Streamlit app will also send raw values. Encoding them here would cause
# inconsistency between training and serving.

# Define features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratified split to preserve class imbalance
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save prepared datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Dropped columns: CustomerID, Unnamed: 0 (if present).")
print("Target distribution (ProdTaken):")
print(y.value_counts(normalize=True))
