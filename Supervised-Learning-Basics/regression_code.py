import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
def run_regression(X, y, name):
    print("\n" + "="*50)
    print(name)
    print("="*50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    best_score = -1
    best_lambda = 0
    for l in [0.01, 0.1, 1, 10, 100]:
        model = Ridge(alpha=l)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_lambda = l
    ridge = Ridge(alpha=best_lambda)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    print("\n--- Ridge Regression ---")
    print("Best Lambda:", best_lambda)
    print("R2:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    best_score = -1
    best_lambda = 0
    for l in [0.01, 0.1, 1, 10, 100]:
        model = Lasso(alpha=l, max_iter=5000)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        score = np.mean(scores)

        if score > best_score:
            best_score = score
            best_lambda = l

    lasso = Lasso(alpha=best_lambda, max_iter=5000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)

    print("\n--- Lasso Regression ---")
    print("Best Lambda:", best_lambda)
    print("R2:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
# =========================================================
# DATASET 1: DB,re.csv
# =========================================================
df1 = pd.read_csv(r'datasets/regression/DB,re.csv')
X1 = df1.iloc[:, 1:-1]
y1 = df1.iloc[:, -1]
X1 = X1.select_dtypes(include=[np.number])
X1 = X1.dropna()
y1 = y1.loc[X1.index]
run_regression(X1, y1, "Dataset 1: DB,re.csv")
# =========================================================
# DATASET 2: DB,re2.csv
# =========================================================
df2 = pd.read_csv(r'datasets/regression/DB,re2.csv')
X2 = df2.drop('charges', axis=1)
y2 = df2['charges']
X2['sex'] = X2['sex'].map({'male':1, 'female':0})
X2['smoker'] = X2['smoker'].map({'yes':1, 'no':0})
X2 = pd.get_dummies(X2, columns=['region'], drop_first=True)

run_regression(X2, y2, "Dataset 2: DB,re2.csv")