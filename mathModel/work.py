# 读取excel文件
import pandas as pd
df = pd.read_excel('data3.xlsx')
df.head()
df.shape[0]

# 先提取出pH，温度，吸附量，ROX率
data1 = df[['pH', '温度', '吸附量','ROX率']]
data1.head()

# 将数据分为训练集和验证集
from sklearn.model_selection import train_test_split
X = data1[['pH', '温度', '吸附量']]
y = data1['ROX率']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 假设的回归方程 Q（去除率）=aX（温度）+bY（剂量）+cZ（pH值）
from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train, y_train)
# print('回归系数：', model.coef_)
# print('截距：', model.intercept_)
# print('训练集的R方：', model.score(X_train, y_train))
# print('验证集的R方：', model.score(X_test, y_test))
# # 线性模型，查看结果好不好
# from sklearn.metrics import mean_squared_error, r2_score
# y_pred = model.predict(X_test)
# print('均方误差：', mean_squared_error(y_test, y_pred))
# print('R方：', r2_score(y_test, y_pred))


X = data1[['pH', '温度', '吸附量']]
y = data1['ROX率']
# 机器学习，获取，线性回归效果差，使用三元多次方程来做
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
print('回归系数：', model.coef_)
print('截距：', model.intercept_)
print('训练集的R方：', model.score(X_poly, y))
print('验证集的R方：', model.score(X_poly, y))
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_poly)
print('均方误差：', mean_squared_error(y, y_pred))
print('R方：', r2_score(y, y_pred))


# ... 上面的代码省略 ...

# 获取模型系数
coefficients = model.coef_
intercept = model.intercept_

# 打印方程
equation = f"y = {intercept:.2f}"
for i, coef in enumerate(coefficients):
    # 根据多项式的顺序，确定特征名称
    feature_name = poly.get_feature_names_out(input_features=['pH', '温度', '吸附量'])[i]
    equation += f" + {coef:.2f}{feature_name}"

print(equation)

# 绘画出上面式子的图像