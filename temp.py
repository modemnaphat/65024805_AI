import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

File_Path = 'C:/65024805/se/'
File_Name = 'car_data.csv'

df = pd.read_csv(File_Path + File_Name)

encoders = []
df.drop(columns=['User ID'], inplace=True)
for i in range(0, len(df.columns) -1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
    encoders.append(enc)

x = df.iloc[:, 0:3]
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x, y)

x_pred = []
for i in range(0, len(df.columns) -1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
x_pred_adj  = np.array(x_pred).reshape(-1, 3)

y_pred = model.predict(x_pred_adj)
print('Pre', y_pred[0])
score = model.score(x, y)
print('Acc', '{:.2f}'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25, 20))
_ = plot_tree(model,
              feature_names=feature,
              class_names=Data_class,
              label='all',
              impurity=True,
              precision= 3,
              filled= True,
              rounded= True,
              fontsize=16)

plt.show()