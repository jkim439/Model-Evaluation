__author__ = 'Junghwan Kim'
__copyright__ = 'Copyright 2016-2019 Junghwan Kim. All Rights Reserved.'
__version__ = '1.0.0'


import matplotlib.pyplot as plt
import numpy as np

from cycler import cycler
from inspect import signature
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils.multiclass import unique_labels


# score
def scoring(y_true, y_pred):
    if (len([i for i in list(set(y_true)) if i > 1]) is 0) and (len([i for i in list(set(y_true)) if i < 2]) is 2):
        binary = True
    else:
        binary = False

    print('\n- Classification Scoring -')

    # Accuracy
    print('Accuracy:', round(accuracy_score(y_true, y_pred), 2))

    # Balanced Accuracy
    print('Balanced Accuracy:', round(balanced_accuracy_score(y_true, y_pred), 2))

    # Average Precision (only for binary class)
    if binary is True:
        print('Average Precision Micro:', round(average_precision_score(y_true, y_pred, average='micro'), 2))
        print('Average Precision Macro:', round(average_precision_score(y_true, y_pred, average='macro'), 2))
        print('Average Precision Weighted:', round(average_precision_score(y_true, y_pred, average='weighted'), 2))
        print('Average Precision Samples:', round(average_precision_score(y_true, y_pred, average='samples'), 2))

    # F1
    if binary is True:
        print('F1 Binary:', round(f1_score(y_true, y_pred, average='binary'), 2))
        print('F1 Samples:', round(f1_score(y_true, y_pred, average='samples'), 2))
    if binary is False:
        print('F1 Micro:', round(f1_score(y_true, y_pred, average='micro'), 2))
        print('F1 Macro:', round(f1_score(y_true, y_pred, average='macro'), 2))
        print('F1 Weighted:', round(f1_score(y_true, y_pred, average='weighted'), 2))

    # Precision
    if binary is True:
        print('Precision Binary:', round(precision_score(y_true, y_pred, average='binary'), 2))
        print('Precision Samples:', round(precision_score(y_true, y_pred, average='samples'), 2))
    if binary is False:
        print('Precision Micro:', round(precision_score(y_true, y_pred, average='micro'), 2))
        print('Precision Macro:', round(precision_score(y_true, y_pred, average='macro'), 2))
        print('Precision Weighted:', round(precision_score(y_true, y_pred, average='weighted'), 2))

    # Recall
    if binary is True:
        print('Recall Binary:', round(recall_score(y_true, y_pred, average='binary'), 2))
        print('Recall Samples:', round(recall_score(y_true, y_pred, average='samples'), 2))
    if binary is False:
        print('Recall Micro:', round(recall_score(y_true, y_pred, average='micro'), 2))
        print('Recall Macro:', round(recall_score(y_true, y_pred, average='macro'), 2))
        print('Recall Weighted:', round(recall_score(y_true, y_pred, average='weighted'), 2))

    # Jaccard
    if binary is True:
        print('Jaccard Binary:', round(jaccard_score(y_true, y_pred, average='binary'), 2))
        print('Jaccard Samples:', round(jaccard_score(y_true, y_pred, average='samples'), 2))
    if binary is False:
        print('Jaccard Micro:', round(jaccard_score(y_true, y_pred, average='micro'), 2))
        print('Jaccard Macro:', round(jaccard_score(y_true, y_pred, average='macro'), 2))
        print('Jaccard Weighted:', round(jaccard_score(y_true, y_pred, average='weighted'), 2))

    # ROC Auc (only for binary class)
    if binary is True:
        print('ROC Auc Samples:', round(roc_auc_score(y_true, y_pred, average='samples'), 2))
        print('ROC Auc Micro:', round(roc_auc_score(y_true, y_pred, average='micro'), 2))
        print('ROC Auc Macro:', round(roc_auc_score(y_true, y_pred, average='macro'), 2))
        print('ROC Auc Weighted:', round(roc_auc_score(y_true, y_pred, average='weighted'), 2))


# confusion_matrix
def matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = np.array(classes)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


# classification_report
def report(y_true, y_pred, classes):
    print('\n- Classification Report -')
    print(classification_report(y_true, y_pred, target_names=classes))


# precision_recall_curve
def recall(y_true, y_scores):
    average_precision = average_precision_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


# roc_curve
def roc(y_true, y_scores, y_class, classes):
    y_test = np.array(y_true)
    y_score = np.array(y_scores)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 3

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(y_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= y_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.rc('axes', prop_cycle=(
        cycler('color', ['red', 'gold', 'green', 'blue', 'purple', 'orange', 'lightgreen', 'cyan', 'magenta'])))

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             linestyle=':', linewidth=4)

    for i in range(y_class):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

