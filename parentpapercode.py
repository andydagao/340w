import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

#load dataset
df = pd.read_csv("data/maindataset.csv")

#create binary target
df["food_insecure"] = ((df["Sometimes not enough to eat"].fillna(0) +
                        df["Often not enough to eat"].fillna(0)) > 0).astype(int)


#drop irrelevant columns
drop_cols = [
    "Enough of the kinds of food wanted",
    "Enough Food, but not always the kinds wanted",
    "Sometimes not enough to eat", 
    "Often not enough to eat", 
    "Did not report", 
    "week_name", "Year", "Location"
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

#fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

#encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Split features and target
X = df.drop("food_insecure", axis=1)
y = df["food_insecure"]

#split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train random forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

#plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances for Predicting Food Insecurity")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()