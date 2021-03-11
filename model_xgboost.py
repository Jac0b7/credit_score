import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

pd.set_option('display.max_rows', None, 'display.width', None)

random_state = 77

# Data Balancing:
print('Balance Data')
data = pd.read_pickle('clean_cs-data.pkl')
X = data.drop('DlqIn2Years', axis=1, inplace=False).copy()
y = data['DlqIn2Years'].copy()

sm = SMOTE(random_state=random_state)
X, y = sm.fit_resample(X, y)

# Train/Test:
print('Split Data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=random_state)


D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

# Model:
print('Learning')
param = {
    'eta': 0.3,
    'max_depth': 15,
    'num_class': 3,
    }

steps = 20

model = xgb.train(param, D_train, steps)
# model = RandomForestClassifier(n_estimators=100, random_state=random_state)
fcast = model.predict(D_test)


# Validation:
print('Validation')
acc = accuracy_score(y_test, fcast)
rec = recall_score(y_test, fcast)
prec = precision_score(y_test, fcast)
cm = confusion_matrix(y_test, fcast)
auc = roc_auc_score(y_test, fcast)

print(f'Accuracy: {acc}')
print(f'Recall: {rec}')
print(f'Precision: {prec}')
print(f'AUC: {auc}')
plt.figure(figsize=(8, 6))
plt.title('XgBoost Confusion Matrix', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()

fpr, tpr, _ = roc_curve(y_test, fcast)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='navy',
         lw=lw, label=f'XgBoost ROC curve (area = {auc})')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
