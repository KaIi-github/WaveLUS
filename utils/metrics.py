import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def plot_roc_auc(y_true, y_prob, label_mapping, first_pth, title, figsize, save_path):
    """
    Plot Multi-class ROC Curves and AUC Values
    
    参数:
        y_true: Grount Truth (n_samples,)
        y_prob: Predictive Probability (n_samples, n_classes)
        label_mapping: Mapping Dictionary {0: 'class1', 1: 'class2', ...}
        figsize: Image Size
    """
    model_name = os.path.basename(first_pth).replace('.pth', '')

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    assert y_prob.ndim == 2, f"y_prob must be a 2D array, the current shape: {y_prob.shape}"
    assert len(y_true) == len(y_prob), f"Sample num mismatch: {len(y_true)} vs {len(y_prob)}"
    assert set(y_true).issubset(label_mapping.keys()), "There exist unmapped label"
    
    # # Binarized labels
    n_classes = len(label_mapping)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Compute the ROC curve
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        fpr[i] = np.insert(fpr[i], 0, 0.0)
        tpr[i] = np.insert(tpr[i], 0, 0.0)
        fpr[i] = np.append(fpr[i], 1.0)
        tpr[i] = np.append(tpr[i], 1.0)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    all_fpr = np.unique(np.concatenate([all_fpr, [0.0, 1.0]]))
    all_fpr.sort()
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # plotting
    plt.figure(figsize=figsize)
    
    specificity_macro = 1 - fpr["macro"]
    plt.plot(specificity_macro, tpr["macro"], 
             label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
             color='darkblue', linewidth=3)
    
    if len(specificity_macro) > 0 and len(tpr["macro"]) > 0:
        first_spec = specificity_macro[0]
        first_tpr = tpr["macro"][0]
        if first_spec == 1.0 and first_tpr != 0.0:
            plt.plot([1.0, 1.0], [0.0, first_tpr], 
                     color='darkblue', 
                     linewidth=3,
                     linestyle='-')
    
    plt.gca().invert_xaxis()
    
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.xlim([1.01, -0.01])
    plt.ylim([0.0, 1.05])
    plt.title(f'{title} Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(f'{save_path}/attn{model_name}_voting_results_ROC_Curve_2_{title}.png', 
                    dpi=300, 
                    bbox_inches='tight')
        print(f"The ROC curve has been saved to: {save_path}")
    plt.close()

    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

def roc_auc(y_true, y_prob, label_mapping):
    """
    Plot Multi-class ROC Curves and AUC Values
    
    参数:
        y_true: Grount Truth (n_samples,)
        y_prob: Predictive Probability (n_samples, n_classes)
        label_mapping: Mapping Dictionary {0: 'class1', 1: 'class2', ...}
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    assert y_prob.ndim == 2, f"y_prob must be a 2D array, the current shape: {y_prob.shape}"
    assert len(y_true) == len(y_prob), f"Sample num mismatch: {len(y_true)} vs {len(y_prob)}"
    assert set(y_true).issubset(label_mapping.keys()), "There exist unmapped label"
    
    # # Binarized labels
    n_classes = len(label_mapping)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Compute the ROC curve
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        fpr[i] = np.insert(fpr[i], 0, 0.0)
        tpr[i] = np.insert(tpr[i], 0, 0.0)
        fpr[i] = np.append(fpr[i], 1.0)
        tpr[i] = np.append(tpr[i], 1.0)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    all_fpr = np.unique(np.concatenate([all_fpr, [0.0, 1.0]]))
    all_fpr.sort()
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc["macro"]

def calculate_metrics(y_true, y_pred, y_prob, label_mapping, average_type='macro'):
    """
    Compute metrics for multi-class classification tasks
    Parameters:
        y_true: Grount Truth (n_samples,)
        y_pred: Predictive class (n_samples,)
        y_prob: Predictive Probability (n_samples, n_classes)
        average_type: 'macro' or 'weighted'
    Return:
        Metrics Dictionary
    """
    metrics = {}

    # BASE  METRICS
    metrics['Acc'] = accuracy_score(y_true, y_pred)
    metrics['BACC'] = balanced_accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average=average_type)
    metrics['Sensitivity'] = recall_score(y_true, y_pred, average=average_type)  # Sensitivity=Recall
    metrics['F1_Macro'] = f1_score(y_true, y_pred, average=average_type)
    metrics['F1_Weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)

    # Specificity 
    def specificity_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        spec = []
        for i in range(cm.shape[0]):
            tn = np.sum(np.delete(cm, i, axis=0)[:, np.delete(np.arange(cm.shape[0]), i)])
            fp = np.sum(cm[i, np.delete(np.arange(cm.shape[0]), i)])
            spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
        return np.mean(spec)

    metrics['Specificity'] = specificity_score(y_true, y_pred)

    if y_prob is not None:
        try:
            metrics['AUC'] = roc_auc(y_true, y_prob, label_mapping)
        except:
            metrics['AUC'] = np.nan

    metrics['Class-wise Report'] = {
        'Precision': precision_score(y_true, y_pred, average=None),
        'Recall': recall_score(y_true, y_pred, average=None),
        'F1': f1_score(y_true, y_pred, average=None),
    }

    return metrics

def calculate_specificity(y_true, y_pred, num_classes):
    """
    Calculate the specificity for each class in multi-class classification tasks.

    Parameters:
        y_true (list or np.array): Grount Truth (n_samples,)
        y_pred (list or np.array): Predictive class (n_samples,)
        num_classes (int): class number

    Return:
        specificities (list): Specificity for each class
    """
    specificities = []
    for class_idx in range(num_classes):
        y_true_binary = np.array(y_true) == class_idx
        y_pred_binary = np.array(y_pred) == class_idx

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        specificity = tn / (tn + fp)
        specificities.append(specificity)
    return specificities



def print_metrics(blinenum_metrics, plinestate_metrics, output_file_path):
    with open(output_file_path, 'a') as f:
        #  Aggregate Metrics
        f.write(f"============ Aggregate Metrics =============\n")
        f.write(f"{'Metric':<20}{'Blinenum':<10}{'Plinestate':<10}\n")
        for k in blinenum_metrics.keys():
            if not isinstance(blinenum_metrics[k], dict):
                f.write(f"{k:<20}{blinenum_metrics[k]:<10.4f}{plinestate_metrics[k]:<10.4f}\n")
        f.write(f"========== ====================== ==========\n")

        #  Class-wise Details
        f.write(f"============ Class-wise Details ============\n")
        f.write(f"{'Class':<10}{'Metric':<15}{'Blinenum':<10}{'Plinestate':<10}\n")
        for i in range(4):
            if i <= 3:
                f.write(f"---------- ---------------------- ----------\n")
            f.write(f"{'Class '+str(i):<10}{'Precision':<15}{blinenum_metrics['Class-wise Report']['Precision'][i]:<10.4f}{plinestate_metrics['Class-wise Report']['Precision'][i]:<10.4f}\n")
            f.write(f"{'Class '+str(i):<10}{'Recall':<15}{blinenum_metrics['Class-wise Report']['Recall'][i]:<10.4f}{plinestate_metrics['Class-wise Report']['Recall'][i]:<10.4f}\n")
            f.write(f"{'Class '+str(i):<10}{'F1':<15}{blinenum_metrics['Class-wise Report']['F1'][i]:<10.4f}{plinestate_metrics['Class-wise Report']['F1'][i]:<10.4f}\n")
        f.write(f"========== ====================== ==========\n")


# e.g.
if __name__ == "__main__":
    y_true = np.random.randint(0, 4, size=1000) 
    y_pred = np.random.randint(0, 4, size=1000)
    y_prob = np.random.rand(1000, 4)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    print(y_true.shape)
    print(y_pred.shape)
    print(y_prob.shape)

    metrics = calculate_metrics(y_true, y_pred, y_prob, average_type='macro')