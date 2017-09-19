# TODO
# How to calculate precision, recall, fscore - when the denominator are 0.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def metrics(labels, probs, threshold):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for x, y in zip(probs, labels):
        guess = 0
        if x >= threshold:
            guess = 1
        if guess == 1 and guess == y:
            true_positive += 1
        elif guess == 0 and guess == y:
            true_negative += 1
        elif guess == 1 and guess != y:
            false_positive += 1
        elif guess == 0 and guess != y:
            false_negative += 1
    return true_positive, true_negative, false_positive, false_negative


def precision(true_positive, false_positive, true_negative, false_negative):
    if (true_positive + false_positive) == 0:
        return 0
    return true_positive / float(true_positive + false_positive)


def recall(true_positive, false_positive, true_negative, false_negative):
    if (true_positive + false_negative) == 0:
        return 0
    return true_positive / float(true_positive + false_negative)


# Sensitivity is the proportion of patients with disease who test positive.
def sensitivity(true_positive, false_positive, true_negative, false_negative):
    return true_positive / float(true_positive + false_negative)


# Specificity is the proportion of patients without disease who test negative.
def specificity(true_positive, false_positive, true_negative, false_negative):
    return true_negative / float(false_positive + true_negative)


def true_positive_rate(true_positive, false_positive, true_negative, false_negative):
    return true_positive / float(true_positive + false_negative)


def false_positive_rate(true_positive, false_positive, true_negative, false_negative):
    return false_positive / float(false_positive + true_negative)


def fscore(true_positive, false_positive, true_negative, false_negative):
    p = precision(true_positive, false_positive, true_negative, false_negative)
    r = recall(true_positive, false_positive, true_negative, false_negative)
    if (p + r) == 0:
        return 0
    return 2 * p * r / float(p + r)


def find_max_index(scores):
    m = max(scores)
    return [i for i, j in enumerate(scores) if j == m]


def find_min_index(scores):
    m = min(scores)
    return [i for i, j in enumerate(scores) if j == m]


def plot_curve(fpr, tpr, auc_roc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def optimalROCPoint(labels, probs, thresholds, condition):
    precision_scores = []
    recall_scores = []
    f_scores = []
    tprs = []
    fprs = []
    sensitivity_scores = []
    specificity_scores = []
    reqd_indices = []
    for threshold in thresholds:
        print "------------------ threshold : ", threshold, " -----------------------"
        true_positive, true_negative, false_positive, false_negative = metrics(labels, probs, threshold)
        precision_score = precision(true_positive, false_positive, true_negative, false_negative)
        recall_score = recall(true_positive, false_positive, true_negative, false_negative)
        f_score = fscore(true_positive, false_positive, true_negative, false_negative)
        tpr = true_positive_rate(true_positive, false_positive, true_negative, false_negative)
        fpr = false_positive_rate(true_positive, false_positive, true_negative, false_negative)
        sensitivity_score = sensitivity(true_positive, false_positive, true_negative, false_negative)
        specificity_score = specificity(true_positive, false_positive, true_negative, false_negative)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f_scores.append(f_score)
        tprs.append(tpr)
        fprs.append(fpr)
        sensitivity_scores.append(sensitivity_score)
        specificity_scores.append(specificity_score)
        print "TP : ", true_positive
        print "FP : ", false_positive
        print "TN : ", true_negative
        print "FN : ", false_negative
        print "precision : ", precision_score
        print "recall : ", recall_score
        print "sensitivity : ", sensitivity_score
        print "specificity : ", specificity_score
        print "tpr : ", tpr
        print "fpr : ", fpr
        print "fscore : ", f_score
        print

    if condition == "fscore":
        reqd_indices = find_max_index(f_scores)
    elif condition == "precision":
        reqd_indices = find_max_index(precision_scores)
    elif condition == "recall":
        reqd_indices = find_max_index(recall_scores)
    elif condition == "sensitivity":
        reqd_indices = find_max_index(sensitivity_scores)
    elif condition == "specificity":
        reqd_indices = find_max_index(specificity_scores)
    elif condition[0] == "tpr":
        max_fpr = condition[1]
        filtered_indices = [i for i, j in enumerate(fprs) if j <= max_fpr]
        temp_tprs = [-1 for i in tprs]
        for index in filtered_indices:
            temp_tprs[index] = tprs[index]
        reqd_indices = find_max_index(temp_tprs)
    elif condition[0] == "fpr":
        min_tpr = condition[1]
        filtered_indices = [i for i, j in enumerate(tprs) if j >= min_tpr]
        temp_fprs = [2 for i in fprs]
        for index in filtered_indices:
            temp_fprs[index] = fprs[index]
        reqd_indices = find_min_index(temp_fprs)
    auc_roc = auc(fprs, tprs)
    plot_curve(fprs, tprs, auc_roc)
    optimal_thresholds = [thresholds[i] for i in reqd_indices]
    print "condition : ", condition, " optimal_thresholds : ", optimal_thresholds
    return optimal_thresholds


def main():
    probs_file = pd.read_csv('HW1probs.csv')
    labels_file = pd.read_csv('HW1Labels.csv')

    probs = np.array(probs_file.x)
    labels = np.array(labels_file.x)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_roc = auc(fpr, tpr)
    print auc_roc  # 0.793770848777
    plot_curve(fpr, tpr, auc_roc)

    threshold = 0.5
    true_positive, true_negative, false_positive, false_negative = metrics(labels, probs, threshold)

    precision_score = precision(true_positive, false_positive, true_negative, false_negative)
    recall_score = recall(true_positive, false_positive, true_negative, false_negative)
    f_score = fscore(true_positive, false_positive, true_negative, false_negative)
    sensitivity_score = sensitivity(true_positive, false_positive, true_negative, false_negative)
    specificity_score = specificity(true_positive, false_positive, true_negative, false_negative)
    print precision_score  # 0.575129533679
    print recall_score  # 0.182565789474
    print f_score  # 0.277153558052
    print sensitivity_score  # 0.182565789474
    print specificity_score  # 0.973751600512

    thresholds = np.arange(0, 1, 0.1)
    #optimalROCPoint(labels, probs, thresholds, "precision")
    #optimalROCPoint(labels, probs, thresholds, "fscore")
    #optimalROCPoint(labels, probs, thresholds, "recall")
    optimalROCPoint(labels, probs, thresholds, ["tpr", 0.9])
    optimalROCPoint(labels, probs, thresholds, ["fpr", 0.6])


if __name__ == "__main__":
    main()
