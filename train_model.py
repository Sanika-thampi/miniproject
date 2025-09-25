import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#Data collection
df=pd.read_csv("retail_store_inventory.csv")
df.info()

# Step 1: Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Step 2: Extract year, month, day
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day

# Step 3: Create weekend flag (1 = weekend, 0 = weekday)
df["weekend"] = df["Date"].dt.weekday.isin([5, 6]).astype(int)

# Preview result
print(df.head())


#slecting required features
features=['Store ID','Product ID','year','month']
targets=['Inventory Level','Units Ordered','Units Sold','Price']
X=df[features]
y=df[targets]

# 6. Split Data into Training and Testing Sets
# ---------------------------------------------
# The dataset is divided into:
# - Training set (80%) → used to train the model
# - Testing set (20%)  → used to evaluate how well the model performs on unseen data

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,     #20% for testing,80% for training
    random_state=42
)
#to print
print("Total rows:",len(X))
print("Training rows:",len(X_train))
print("Testing rows:",len(X_test))

#Train the model using random forest model
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#predict and evaluate
y_pred=model.predict(X_test)

with open("inventory_demand_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(" Model saved successfully as inventory_demand_model.pkl")
