import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_curve,auc
from imblearn.over_sampling import SMOTE
df = pd.read_csv("creditcard.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nInfo")
print(df.info())

print("\nStatistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True),inplace=True)
df.fillna(df.mode().iloc[0],inplace=True)

print("\nClass Distribution")
print(df["Class"].value_counts())

sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

df.hist(figsize=(15, 12))
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df.drop("Class", axis=1)
Y = df["Class"]

x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.2,stratify=Y,random_state=42)
scaler=StandardScaler()

x_train["Amount"] = scaler.fit_transform(x_train[["Amount"]])
x_test["Amount"] = scaler.transform(x_test[["Amount"]])

print("\nBefore SMOTE")
print(y_train.value_counts())
smote=SMOTE(random_state=42)
x_train,y_train=smote.fit_resample(x_train,y_train)

print("\nAfter SMOTE")
print(y_train.value_counts())

models = {"Logistic Regression": LogisticRegression(class_weight="balanced",max_iter=1000),"Random Forest": RandomForestClassifier(class_weight="balanced",random_state=42,n_estimators=100)}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = f1_score(y_test, y_pred)
    results[name] = score
    print(f"\n{name}")
    print(f"F1 Score : {score:.4f}")

best_model_name = max(results,key=results.get)

print("\nBest Model :", best_model_name)
print("Best F1 Score :", results[best_model_name])

model = models[best_model_name]
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
cm = confusion_matrix(y_test,y_pred)

print("\nConfusion Matrix")
print(cm)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report")
print(classification_report(y_test,y_pred))

y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test,y_prob)
roc_auc = auc(fpr,tpr)
print("\nROC-AUC Score :", roc_auc)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("\nModel Saved Successfully!")
