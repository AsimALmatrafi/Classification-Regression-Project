import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================================================
# Function لتقييم الموديلات (نفس الفكرة بس مرتبة)
# =========================================================
def run_models(X, y, name):

    print("\n" + "="*50)
    print(name)
    print("="*50)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ===========================
    # Logistic Regression
    # ===========================
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print("\n--- Logistic Regression ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ===========================
    # Neural Network
    # ===========================
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)

    print("\n--- Neural Network ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# =========================================================
# DATASET 1: Titanic
# =========================================================
df1 = pd.read_csv(r'datasets/classification/DB,class1.csv')

y1 = df1['Survived']
X1 = df1.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1)

X1['Sex'] = X1['Sex'].map({'male':0, 'female':1})
X1['Embarked'] = X1['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2})
X1['Age'] = X1['Age'].fillna(X1['Age'].median())

run_models(X1, y1, "Titanic Dataset")


# =========================================================
# DATASET 2: Iris
# =========================================================
df2 = pd.read_csv(r'datasets/classification/DB,class2.csv')

X2 = df2.drop('Species', axis=1)
y2 = df2['Species']

le = LabelEncoder()
y2 = le.fit_transform(y2)

run_models(X2, y2, "Iris Dataset")


# =========================================================
# DATASET 3: Wine Quality
# =========================================================
df3 = pd.read_csv(r'datasets/classification/DB,class3.csv')

X3 = df3.drop('quality', axis=1)
y3 = df3['quality']

run_models(X3, y3, "Wine Quality Dataset")


# =========================================================
# DATASET 4: Heart Disease
# =========================================================
df4 = pd.read_csv(r'datasets/classification/DB,class4.csv')

X4 = df4.drop('HeartDisease', axis=1)
y4 = df4['HeartDisease']

# encoding
X4['Sex'] = X4['Sex'].map({'M':0, 'F':1})
X4['ChestPainType'] = X4['ChestPainType'].map({'ATA':0,'NAP':1,'ASY':2,'TA':3})
X4['RestingECG'] = X4['RestingECG'].map({'Normal':0,'ST':1,'LVH':2})
X4['ExerciseAngina'] = X4['ExerciseAngina'].map({'N':0,'Y':1})
X4['ST_Slope'] = X4['ST_Slope'].map({'Up':0,'Flat':1,'Down':2})

# fill missing
for col in X4.columns:
    X4[col] = X4[col].fillna(X4[col].median())

run_models(X4, y4, "Heart Disease Dataset")