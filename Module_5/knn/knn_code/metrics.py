from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def print_scores(y_true, y_pred, y_proba, test = True):
    
    """
    Prints out accuracy, recall, precision, and f1_score
    for either train or test sets.
    """
   
    if test == True:
        print(f"Test accuracy:  {accuracy_score(y_pred, y_true)}")
        print(f"Test recall: {recall_score(y_pred, y_true)}")
        print(f"Test precision: {precision_score(y_pred, y_true)}")
        print(f"Test f1: {f1_score(y_pred, y_true)}")
        print(f"ROC_AUC: {roc_auc_score(y_true, y_proba[:,1])}")
        
    else:
        print(f"Train accuracy:  {accuracy_score(y_pred, y_true)}")
        print(f"Train recall: {recall_score(y_pred, y_true)}")
        print(f"Train precision: {precision_score(y_pred, y_true)}")
        print(f"Train f1: {f1_score(y_pred, y_true)}")
        print(f"ROC_AUC: {roc_auc_score(y_true, y_proba[:,1])}")