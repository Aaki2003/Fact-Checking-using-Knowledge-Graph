import numpy as np 
import pandas as pd 
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

transEdf = pd.read_csv('/kaggle/working/transE_Predictions.csv')
transEdf.head()

res = transEdf['Result'].iloc[1]
print(res)

res = transEdf['Result'].iloc[2]
print(res)

true_list = ['true',"'true'", "'true'."]
false_list = ['false', "'false'", "'false'."]
nei_list = ['nei', "'nei'",  "'nei'."]
for index, row in transEdf.iterrows():
    for word in reversed(row['Result'].split()):
        word = word.lower()
        if word in true_list or word in false_list or word in nei_list:
            if(word in true_list):
                transEdf.at[index, 'Prediction'] = 'supported'
            if(word in false_list):
                transEdf.at[index, 'Prediction'] = 'refuted'
            if(word in nei_list):
                transEdf.at[index, 'Prediction'] = 'NEI'
            break
    else:
        transEdf.at[index, 'Prediction'] = 'NEI'



from sklearn.metrics import precision_score, recall_score, f1_score
y_true = transEdf['cleaned_truthfulness']
y_pred = transEdf['Prediction']

valid_labels = ["supported", "refuted", "NEI"]


# Calculate precision, recall, and F1-score for each label
precision = precision_score(y_true, y_pred, labels=valid_labels, average=None)
recall = recall_score(y_true, y_pred, labels=valid_labels, average=None)
f1 = f1_score(y_true, y_pred, labels=valid_labels, average=None)
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Display per-class metrics
metrics_df = pd.DataFrame({
    "Label": valid_labels,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})
print(metrics_df)

# Calculate overall metrics
macro_precision = precision_score(y_true, y_pred, labels=valid_labels, average='macro')
macro_recall = recall_score(y_true, y_pred, labels=valid_labels, average='macro')
macro_f1 = f1_score(y_true, y_pred, labels=valid_labels, average='macro')

print(f"\nMacro Precision: {macro_precision:.2f}")
print(f"Macro Recall: {macro_recall:.2f}")
print(f"Macro F1-Score: {macro_f1:.2f}")

# Weighted metrics (accounts for class imbalance)
weighted_precision = precision_score(y_true, y_pred, labels=valid_labels, average='weighted')
weighted_recall = recall_score(y_true, y_pred, labels=valid_labels, average='weighted')
weighted_f1 = f1_score(y_true, y_pred, labels=valid_labels, average='weighted')

print(f"\nWeighted Precision: {weighted_precision:.2f}")
print(f"Weighted Recall: {weighted_recall:.2f}")
print(f"Weighted F1-Score: {weighted_f1:.2f}")



# Plot metrics
metrics_df.set_index("Label").plot(kind="bar", figsize=(10, 6))
plt.title("Precision, Recall, and F1-Score for Each Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="lower right")
plt.show()
