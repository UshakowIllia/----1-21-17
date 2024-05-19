import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

# Завантаження даних з файлу
df = pd.read_csv('data_metrics.csv')

# Виведення перших декількох рядків для перевірки
print(df.head())

# Встановлення порогу для прогнозування
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
print(df.head())

def find_TP(y_true, y_pred):
    # Кількість True Positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    # Кількість False Negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    # Кількість False Positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    # Кількість True Negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

# Перевірка результатів
print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

def find_conf_matrix_values(y_true,y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true,y_pred)
    FN = find_FN(y_true,y_pred)
    FP = find_FP(y_true,y_pred)
    TN = find_TN(y_true,y_pred)
    return TP,FN,FP,TN
def ushakow_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])

ushakow_confusion_matrix(df.actual_label.values, df.predicted_RF.values)

assert np.array_equal(ushakow_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(df.actual_label.values, df.predicted_RF.values) ), 'ushakow_confusion_matrix() is not correct for RF'
assert np.array_equal(ushakow_confusion_matrix(df.actual_label.values, df.predicted_LR.values),confusion_matrix(df.actual_label.values, df.predicted_LR.values) ), 'ushakow_confusion_matrix() is not correct for LR'

accuracy_score(df.actual_label.values, df.predicted_RF.values)

def ushakow_accuracy_score(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return (TP + TN) / (TP + FN + FP + TN) # як по формулі

assert ushakow_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on'
assert ushakow_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Accuracy RF: %.3f'%(ushakow_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f'%(ushakow_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

recall_score(df.actual_label.values, df.predicted_RF.values)
def ushakow_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return TP / (TP + FN) # як по формулі
assert ushakow_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert ushakow_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Recall RF: %.3f'%(ushakow_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f'%(ushakow_recall_score(df.actual_label.values, df.predicted_LR.values)))

precision_score(df.actual_label.values, df.predicted_RF.values)

def ushakow_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return TP / (TP + FP) # як по формулі
assert ushakow_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert ushakow_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Precision RF: %.3f'%(ushakow_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f'%(ushakow_precision_score(df.actual_label.values, df.predicted_LR.values)))

f1_score(df.actual_label.values, df.predicted_RF.values)

def ushakow_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = ushakow_recall_score(y_true, y_pred)
    precision = ushakow_precision_score(y_true, y_pred)
    return 2 * (recall * precision) / (recall + precision) # як по формулі
assert ushakow_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'ushakow_accuracy_score failed on RF'
assert ushakow_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'ushakow_accuracy_score failed on LR'
print('F1 RF: %.3f'%(ushakow_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f'%(ushakow_f1_score(df.actual_label.values, df.predicted_LR.values)))



print('')
print('scores with threshold = 0.5')
print('Accuracy RF: %.3f'%(ushakow_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(ushakow_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(ushakow_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(ushakow_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(ushakow_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(ushakow_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(ushakow_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(ushakow_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Розділення на ознаки (X) та цільову змінну (y)
X = df.drop(columns=['actual_label'])
y = df['actual_label']

# Створення моделі SVM
svm_model = SVC()

# Навчання моделі
svm_model.fit(X, y)

# Оцінка результатів моделі
y_pred_svm = svm_model.predict(X)

 # Оцінка точності
accuracy_svm = accuracy_score(y, y_pred_svm)

# Оцінка повноти
recall_svm = recall_score(y, y_pred_svm)

# Оцінка точності
precision_svm = precision_score(y, y_pred_svm)

# Оцінка F1-оцінки
f1_svm = f1_score(y, y_pred_svm)



print('Accuracy SVM:', accuracy_svm)
print('Recall SVM:', recall_svm)
print('Precision SVM:', precision_svm)
print('F1 Score SVM:', f1_svm)