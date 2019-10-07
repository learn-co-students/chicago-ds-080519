from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

def log_scores(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred_train = lr.predict(X_train)
    pred_test = lr.predict(X_test)
    prediction_probs = lr.predict_proba(X_test)[:,1]
    
    print(f"Precision: {precision_score(pred_test, y_test)}")
    print(f"Recall: {recall_score(pred_test, y_test)}")
    print(f"Accuracy: {accuracy_score(pred_test, y_test)}")
    print(f"F1: {f1_score(pred_test, y_test)}")
    
    roc_auc = roc_auc_score(y_test, prediction_probs)
    
    print(f'roc_auc {roc_auc}')
    
    return [pred_test, prediction_probs]