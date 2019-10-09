from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion(y_true, y_hat):
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_hat, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()