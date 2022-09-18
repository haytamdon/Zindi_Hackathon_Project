import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
import os

def calculate_metrics(model, X ,y, y_pred):
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    return f1, auc

def plot_auc_roc(y, y_pred):
    fpr, tpr, _ = roc_curve(y, y_pred)
    plt.figure(figsize=(16,8) )
    figure = sns.lineplot(fpr,tpr,label="AUC="+str(auc))
    return figure

def confusion_matrix(model, X, y):
    figure = plot_confusion_matrix(model, X, y)
    return figure


