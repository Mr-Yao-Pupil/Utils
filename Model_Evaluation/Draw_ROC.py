import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def computer_every_class(label_list, model_pre_list):
    class_num = len(set(label_list))
    label_list = label_binarize(label_list, classes=[i for i in range(class_num)])  # 标签的独热编码
    model_pre_list = np.array(model_pre_list)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(label_list[:, i], model_pre_list[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, class_num


def computer_result_function1(fpr, tpr, roc_auc, label_list, model_pre_list):
    class_num = len(set(label_list))
    label_list = label_binarize(label_list, classes=[i for i in range(class_num)])  # 标签的独热编码
    model_pre_list = np.array(model_pre_list)
    fpr["micro"], tpr["micro"], _ = roc_curve(label_list.ravel(), model_pre_list.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc, class_num


def computer_result_function2(fpr, tpr, roc_auc, class_num):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= class_num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc, class_num


def draw_function(fpr, tpr, roc_auc, class_num):
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    # 绘制曲线
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))  # 可以把里面的i通过字典映射成标签名字进行显示，可参考TensorFlow版本

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def exm_main(label_list, pre_list):

    fpr, tpr, roc_auc, class_num = computer_every_class(label_list, pre_list)
    fpr, tpr, roc_auc, class_num = computer_result_function2(roc_auc, fpr, tpr, class_num)
    draw_function(fpr, tpr, roc_auc, class_num)
