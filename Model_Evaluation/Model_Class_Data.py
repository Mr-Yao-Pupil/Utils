from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_result(model_pre_list, label_list):
    """
    获取模型对一批结果的混淆矩阵
    :param model_pre_list: list, 模型所有的预测结果
    :param label_list: list, 标签组成的list
    :return: 混淆矩阵， 准确率、召回率等模型数据
    """
    matrix = confusion_matrix(y_true=model_pre_list, y_pred=label_list, labels=[i for i in list(set(label_list))])
    class_report = classification_report(y_true=model_pre_list, y_pred=label_list,
                                         labels=[i for i in list(set(label_list))],
                                         digits=11)
    return matrix, class_report
