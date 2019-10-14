from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve, accuracy_score, recall_score, f1_score, precision_score


def fit_model_and_return_metrics(skf_splits):
    accuracy_list = []
    
    for train_ind, test_ind in skf_splits:
        X_train = X_not_val.iloc[train_ind]
        X_test = X_not_val.iloc[test_ind]

        y_train = y_not_val.iloc[train_ind]
        y_test = y_not_val.iloc[test_ind]

        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        y_hat_proba = clf.predict_proba(X_test)[:, 1]

        accuracy_list.append(accuracy_score(y_test, y_hat))

def fit_model_and_print_metrics(skf_splits,X_not_val, y_not_val, classifier, print_metrics = False):
    clf = classifier
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    
    metric_list = [accuracy_list, precision_list, recall_list,
                    f1_list, roc_auc_list]
    
    for train_ind, test_ind in skf_splits:
        X_train = X_not_val.iloc[train_ind]
        X_test = X_not_val.iloc[test_ind]

        y_train = y_not_val.iloc[train_ind]
        y_test = y_not_val.iloc[test_ind]

        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        y_hat_proba = clf.predict_proba(X_test)[:, 1]

        accuracy_list.append(accuracy_score(y_test, y_hat))
        precision_list.append(precision_score(y_test, y_hat))
        recall_list.append(recall_score(y_test, y_hat))
        f1_list.append(f1_score(y_test, y_hat))
        roc_auc_list.append(roc_auc_score(y_test, y_hat))
    
    
    mean_metrics = [sum(metric)/len(metric) for metric in metric_list]
    mean_metrics_dict = {}
    #Create a metric dictionary for ease of access.
    metric_names = ['accuracy', 'precision', 'recall',
                    'f1', 'roc_auc']
    
    for mean_metric, metric in zip(mean_metrics, metric_names):
        mean_metrics_dict[metric] = mean_metric
    
    if print_metrics == True:
        print(f"accuracy_score: {mean_metrics[0]}")
        print(f"precision_score: {mean_metrics[1]}")
        print(f"recall_score: {mean_metrics[2]}")
        print(f"f1_score: {mean_metrics[3]}")
        print(f"roc_auc_score: {mean_metrics[4]}")
        
        
    with open('best_estimators.txt', 'a') as write_file:
        write_file.write(f'''{str(clf).replace(",", "")},{mean_metrics[0]},{mean_metrics[1]},{mean_metrics[2]},{mean_metrics[3]},{mean_metrics[4]}\n''')
    return mean_metrics_dict