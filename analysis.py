import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score,roc_curve,precision_recall_curve,average_precision_score,roc_auc_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

df = pd.read_csv("Credit_Card_Fraud/creditcard.csv")
print("Rows,Columns:",df.shape)
print(df.dtypes)
print("Missing values: ",df.isnull().sum().sort_values(ascending=True).head(20))
target_col="Class"
print("/n Target Distribution")
print(df[target_col].value_counts())
print("Fraud Detection: ",df[target_col].mean())

plt.figure(figsize=(8,4))
plt.hist(df["Amount"],bins=100)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.show()

if "Time" in df.columns:
    df["Hour"]=df["Time"]//3600%24
    plt.figure(figsize=(10,4))
    df.groupby("Hour").size().plot(kind="bar")
    plt.title("Transaction by Hour")
    plt.show()

plt.figure(figsize=(5,5))
plt.pie(df["Class"].value_counts().values,labels=["Non Fraud","Fraud"],autopct='%1.1f%%',colors=['Blue',"Red"])
plt.title("Fraud vs Non Fraud Transactions")
plt.show()

plt.figure(figsize=(12,8))
corr=df.corr()
plt.imshow(corr,cmap="coolwarm",interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns,rotation=90)
plt.yticks(range(len(corr.columns)),corr.columns)
plt.title("Feature correlation heatmap")
plt.show()

plt.figure(figsize=(8,4))
amount_nonfraud=df[df['Class']==0]['Amount']
amount_fraud=df[df['Class']==1]['Amount']
plt.boxplot([amount_nonfraud, amount_fraud],tick_labels=['non-fraud','fraud'])
plt.yscale('log')
plt.ylabel('Amount')
plt.title("Transaction by Fraud/Non Fraud")
plt.show()

hourly=df.groupby('Hour')['Class'].sum()
plt.figure(figsize=(10,4))
hourly.plot(kind='bar',color='pink')
plt.ylabel("Fraud Proportion")
plt.title("Fraud Rate by Hour of Day")
plt.show()


df_fe=df.copy()
df_fe['Log Amount']=np.log1p(df_fe['Amount'])
global_mean=df_fe["Amount"].mean()
df_fe['amount_to_mean']=df_fe['Amount']/(global_mean+1e-9)
p99=df["Amount"].quantile(.99)
df_fe['high_amount']=(df_fe['Amount']>p99).astype(int)

if 'Hour' in df_fe.columns:
    df_fe['is_night']=df_fe['Hour'].isin([0,1,2,3,4,5,22,23]).astype(int)
else:
    df_fe['is_night']=0

x=df_fe.drop(columns=[target_col])
y=df_fe[target_col].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

numeric_features=x_train.select_dtypes(include=[np.number]).columns.tolist()
scaler=RobustScaler()
scaler.fit(x_train[numeric_features])

def prep(X_df):
    Xc=X_df.copy()
    Xc[numeric_features]=scaler.transform(Xc[numeric_features])
    
    if 'Time' in Xc.columns:
        Xc=Xc.drop(columns=['Time'])
    return Xc
X_train_proc=prep(x_train)
X_test_proc=prep(x_test)
print("Processed data shapes: ",X_train_proc.shape,X_test_proc.shape)

def evaluate_model(model,x_test,y_test,thresh=0.5,show_pr_curve=True):
    if hasattr(model,'predict_proba'):
        y_scores=model.predict_proba(x_test)[:,1]
    elif hasattr(model,'decision_funtion'):
        s=model.decision_function(x_test)
        y_scores=(s-s.min())/(s.max()-s.min()+1e-9)
    else:
        y_scores=model.predict(x_test)
    y_pred=(y_scores>=thresh).astype(int)
    print("Classification report (threshold={}):".format(thresh))
    print(classification_report(y_test,y_pred,digits=4))
    ap=average_precision_score(y_test,y_scores)
    roc=roc_auc_score(y_test,y_scores)
    cm=confusion_matrix(y_test,y_pred)
    print('Averager Precision Score: ',ap)
    print("ROC AUC Score: ",roc)
    print("Confusion matrix: ",cm)
    if show_pr_curve:
        precision,recall,thresholds=precision_recall_curve(y_test,y_scores)
        plt.figure(figsize=(6,4))
        plt.plot(recall,precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f'PR curve (AP={ap:.4f})')
        plt.show()
    return{'ap':ap,'roc':roc,'scores':y_scores}

trained_models={}
lr=LogisticRegression(max_iter=1000,class_weight='balanced',solver='saga')
lr.fit(X_train_proc,y_train)
trained_models['LogisticRegression']=lr
print("Trained Logistic Regression")

rf=RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=-1,random_state=42)
rf.fit(X_train_proc,y_train)
trained_models['RandomForestClassifier']=rf
print("Trained Random Forest")

xgb_clf=xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss',scale_pos_weight=(np.sum(y_train==0)/np.sum(y_train==1)))
xgb.fit(X_train_proc,y_train)
trained_models['XGBClassifier']=xgb_clf
print("Trained XGB Classifier")

results=[]
for name,model in trained_models.items():
    print("\n---Model",name,'---')
    res=evaluate_model(model,X_test_proc,y_test)
    results.append({'model':name,'AP':res['ap'],'ROC-AUC':res['roc']})
res_df=pd.DataFrame(results).sort_values('AP',ascending=False)
res_df

best_model_name=res_df.iloc[0]['model']
model=trained_models[best_model_name]
print("Best model chosen:",best_model_name)
if hasattr(model,'predict_proba'):
    y_scores=model.predict_proba(X_test_proc)[:,1]
else:
    s=model.decision_function(X_test_proc)
    y_scores=(s-s.min())/(s.max()-s.min()+1e-9)

precision,recall,thresholds=precision_recall_curve(y_test,y_scores)
f1_score=2*(precision*recall)/(precision+recall+1e-9)
best_idx=np.nanargmax(f1_score)
best_thresh=thresholds[best_idx] if best_idx<len(thresholds) else 0.5
print("Best threshold on f1 by test: ",best_thresh)

X_all_proc=prep(x)
if hasattr(model,'predict_proba'):
    all_scores=model.predict_proba(X_all_proc)[:,1]
else:
    s=model.decision_function(X_all_proc)
    all_scores=(s-s.min())/(s.max()-s.min()+1e-9)

df_out=df_fe.copy()
df_out['fraud_score']=all_scores
df_out['prep_fraud']=(df_out['fraud_score']>=best_thresh).astype(int)
flagged=df[df['prep_fraud']==1].sort_values('fraud_score',ascending=False)
print("Flagged Transaction count: ",len(flagged))
flagged.head(20).to_csv("Flagged_transactions.csv",index=False)
print("Flagged transactions saved to Flagged_transactions.csv")


 
