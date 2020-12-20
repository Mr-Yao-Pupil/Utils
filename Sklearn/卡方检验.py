from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

iris = load_iris()
data, label = iris.data, iris.target

# 获取四组特征和结果的相关性, 返回两个array, 第一组array表示特征得分, 第二组array返回p值：即该类特征的犯错概率
chiValues = chi2(data, label)
# 选择k个相关性最大的特征特征, fit_transform返回选择出的这些特征
X_new = SelectKBest(chi2, k=2).fit_transform(data, label)
