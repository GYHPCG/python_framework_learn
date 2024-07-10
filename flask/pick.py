from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import pickle

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

# 训练模型
ada_model = AdaBoostClassifier()
ada_model.fit(X, y)

# 保存模型为pkl文件
with open('adaboost_model.pkl', 'wb') as f:
    pickle.dump(ada_model, f)
