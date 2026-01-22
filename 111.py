#!/usr/bin/env python
# coding: utf-8

# # 泰坦尼克号生存预测（一体化版本，无需下载文件）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 设置图表样式
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("泰坦尼克号生存预测 - 自动下载数据版本")
print("=" * 60)

# 1. 自动创建模拟数据（避免下载）
print("\n1. 创建模拟数据集...")

# 训练数据
train_data = pd.DataFrame({
    'PassengerId': range(1, 892),
    'Survived': np.random.choice([0, 1], 891, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], 891, p=[0.24, 0.21, 0.55]),
    'Name': ['John Doe'] * 891,
    'Sex': np.random.choice(['male', 'female'], 891, p=[0.65, 0.35]),
    'Age': np.random.normal(29.7, 14.5, 891).clip(0.42, 80),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], 891, p=[0.68, 0.14, 0.09, 0.04, 0.02, 0.02, 0.01]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], 891, p=[0.76, 0.13, 0.06, 0.03, 0.01, 0.005, 0.005]),
    'Ticket': ['A/5 21171'] * 891,
    'Fare': np.random.exponential(32.2, 891).clip(0, 512.33),
    'Cabin': [None] * 891,
    'Embarked': np.random.choice(['S', 'C', 'Q'], 891, p=[0.72, 0.19, 0.09])
})

# 调整一些模式，使其更接近真实数据
# 女性生存率更高
train_data.loc[train_data['Sex'] == 'female', 'Survived'] = np.random.choice([0, 1], 
    sum(train_data['Sex'] == 'female'), p=[0.26, 0.74])
# 头等舱生存率更高
train_data.loc[train_data['Pclass'] == 1, 'Survived'] = np.random.choice([0, 1], 
    sum(train_data['Pclass'] == 1), p=[0.37, 0.63])
# 儿童生存率更高
train_data.loc[train_data['Age'] < 12, 'Survived'] = np.random.choice([0, 1], 
    sum(train_data['Age'] < 12), p=[0.45, 0.55])

# 测试数据
test_data = pd.DataFrame({
    'PassengerId': range(892, 1310),
    'Pclass': np.random.choice([1, 2, 3], 418, p=[0.24, 0.21, 0.55]),
    'Name': ['Jane Smith'] * 418,
    'Sex': np.random.choice(['male', 'female'], 418, p=[0.65, 0.35]),
    'Age': np.random.normal(30.3, 12.5, 418).clip(0.17, 76),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], 418, p=[0.68, 0.14, 0.09, 0.04, 0.02, 0.02, 0.01]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], 418, p=[0.76, 0.13, 0.06, 0.03, 0.01, 0.005, 0.005]),
    'Ticket': ['PC 17599'] * 418,
    'Fare': np.random.exponential(35.6, 418).clip(0, 512.33),
    'Cabin': [None] * 418,
    'Embarked': np.random.choice(['S', 'C', 'Q'], 418, p=[0.72, 0.19, 0.09])
})

print(f"训练数据形状: {train_data.shape}")
print(f"测试数据形状: {test_data.shape}")

# 2. 数据分析
print("\n2. 数据分析和可视化...")

# 生存比例
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
survived_counts = train_data['Survived'].value_counts()
axes[0].pie(survived_counts.values, labels=['遇难', '幸存'], autopct='%1.1f%%',
           startangle=90, colors=['#ff9999', '#66b3ff'])
axes[0].set_title('乘客生存比例')

# 性别与生存关系
gender_survival = train_data.groupby('Sex')['Survived'].mean()
axes[1].bar(['男性', '女性'], [gender_survival['male'], gender_survival['female']], 
           color=['skyblue', 'lightcoral'])
axes[1].set_ylabel('生存率')
axes[1].set_title('性别与生存率关系')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"整体生存率: {train_data['Survived'].mean():.2%}")
print(f"女性生存率: {gender_survival['female']:.2%}")
print(f"男性生存率: {gender_survival['male']:.2%}")

# 3. 特征工程
print("\n3. 特征工程...")

# 合并数据以便统一处理
combined_data = pd.concat([train_data.drop('Survived', axis=1), test_data], ignore_index=True)
combined_data['Survived'] = pd.concat([train_data['Survived'], pd.Series([0]*len(test_data))], ignore_index=True)

# 创建新特征
combined_data['Title'] = combined_data['Name'].apply(lambda x: 'Mr' if 'Mr' in x else ('Mrs' if 'Mrs' in x else ('Miss' if 'Miss' in x else 'Other')))
combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1
combined_data['IsAlone'] = (combined_data['FamilySize'] == 1).astype(int)

# 编码分类变量
combined_data['Sex'] = combined_data['Sex'].map({'male': 0, 'female': 1})
combined_data['Embarked'] = combined_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

# 处理缺失值
combined_data['Age'] = combined_data['Age'].fillna(combined_data['Age'].median())
combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].median())

# 4. 机器学习建模
print("\n4. 机器学习建模...")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 准备特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X = combined_data[features][:len(train_data)]
y = combined_data['Survived'][:len(train_data)]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"模型准确率: {accuracy:.2%}")
print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")

# 5. 特征重要性
print("\n5. 特征重要性分析...")

feature_importance = pd.DataFrame({
    '特征': features,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['特征'], feature_importance['重要性'], color='steelblue')
plt.xlabel('重要性得分')
plt.title('随机森林特征重要性排名')
plt.gca().invert_yaxis()

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# 6. 预测测试集
print("\n6. 预测测试集并生成结果...")

X_test = combined_data[features][len(train_data):]
test_predictions = model.predict(X_test)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions.astype(int)
})

# 保存结果
submission.to_csv('titanic_predictions.csv', index=False)
print(f"预测完成！结果已保存为 'titanic_predictions.csv'")
print(f"预测生存人数: {submission['Survived'].sum()} (占测试集的 {submission['Survived'].mean():.1%})")

# 7. 结果展示
print("\n7. 结果总结:")
print("=" * 50)
print(f"数据集统计:")
print(f"  - 训练样本: {len(train_data)} 名乘客")
print(f"  - 测试样本: {len(test_data)} 名乘客")
print(f"  - 整体生存率: {train_data['Survived'].mean():.2%}")
print(f"\n模型性能:")
print(f"  - 验证集准确率: {accuracy:.2%}")
print(f"  - 最重要特征: {feature_importance.iloc[0]['特征']}")
print(f"\n预测结果:")
print(f"  - 测试集预测生存人数: {submission['Survived'].sum()}")
print(f"  - 结果文件: titanic_predictions.csv")
print("=" * 50)

# 显示预测结果前10行
print("\n预测结果前10行:")
print(submission.head(10))

# 最终可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 舱位等级与生存率
pclass_survival = train_data.groupby('Pclass')['Survived'].mean()
axes[0].bar(['一等舱', '二等舱', '三等舱'], pclass_survival.values, color=['gold', 'silver', 'brown'])
axes[0].set_ylabel('生存率')
axes[0].set_title('舱位等级与生存率')
axes[0].grid(True, alpha=0.3)

# 2. 年龄分布
axes[1].hist(train_data['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('年龄')
axes[1].set_ylabel('人数')
axes[1].set_title('乘客年龄分布')
axes[1].grid(True, alpha=0.3)

# 3. 预测结果分布
survived_counts = submission['Survived'].value_counts()
axes[2].pie(survived_counts.values, labels=['预测遇难', '预测幸存'], autopct='%1.1f%%',
           startangle=90, colors=['#ff9999', '#66b3ff'])
axes[2].set_title('测试集预测结果分布')

plt.tight_layout()
plt.show()

print("\n✅ 程序运行完成！所有分析、建模和预测已完成。")
