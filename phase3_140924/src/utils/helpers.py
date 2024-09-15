# helpers.py content
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def compute_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return f1, acc, cm
