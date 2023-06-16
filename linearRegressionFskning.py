import numpy as np 
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_hist_gradient_boosting
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
%matplotlib inline

!rm -r machine-learning/  # first remove folder, if present
!git clone https://github.com/Emmsiiii/fskning

dataFrame = pd.read_csv("fskning/Data cleaned.csv", delimiter=";")

dataFrame.head()

dataFrame.shape

dataFrame.corr()

dataFrame.describe()

dataFrame.info()

X = dataFrame.astype(object)
y = dataFrame.astype(object)
y.info()

X = dataFrame[['Alder', 'Køn','CPM (pris pr. 1000 eksponeringer)']]
y = dataFrame['Eksponeringer']

X = pd.get_dummies(X) 
y = pd.get_dummies(y)

dataFrame.describe()

sns.pairplot(dataFrame, x_vars=['CPM (pris pr. 1000 eksponeringer)','Køn','Alder'], y_vars=['Eksponeringer','Alder'], height=4, aspect=1, kind='scatter')
plt.show()

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

y_train.shape
y_test.shape

reg_model = linear_model.LinearRegression()

reg_model = LinearRegression().fit(X_train, y_train)

print('Intercept: ',reg_model.intercept_)

list(zip(X, reg_model.coef_))

y_pred= reg_model.predict(X_test)  
x_pred= reg_model.predict(X_train) 

print("Prediction for test set: {}".format(y_pred))

reg_model_diff = ({'Actual value': y_test, 'Predicted value': y_pred})
reg_model_diff

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)