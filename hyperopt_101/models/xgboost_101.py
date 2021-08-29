import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#Import dataset

df = pd.read_csv("/Users/ruksvaithy/Documents/Projects/hyperopt_101/data//challenge.csv", sep=",")
df = df.fillna(0)
one_hot = pd.get_dummies(df["NOC"])
df = df.drop('NOC',axis = 1)
# Join the encoded df
df = df.join(one_hot)

#split dataset
y = df["Target"]
X = df.drop("Target", axis=1)

print(df.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

Xt =pd.concat([X_train,y_train],axis=1)


def acc_model(params):

    return cross_val_score(clf, X_train, y_train, scoring='neg_root_mean_squared_error').mean()


params = {
    'year': hp.choice('year', range(1984,2011, 4)),
    #'max_depth': hp.choice('max_depth', range(1,20)),
    #'max_features': hp.choice('max_features', range(1,150)),
    'n_estimators': hp.choice('n_estimators', range(0,300)),
    'min_samples_split': hp.choice('min_n', range(0,100)),
    'learning_rate': hp.uniform('learning_rate',0.1,1)}


def f(params):
    print(params)
    clf = xgb.XGBRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    X_t = Xt[Xt['Year']>params['year']]
    y_t = X_t["Target"]
    X_t = X_t.drop("Target", axis=1)
    clf.fit(X_t,y_t)
    preds = clf.predict(X_test)
    mse_scr = metrics.mean_squared_error(y_test, preds)

    print("SCORE:", np.sqrt(mse_scr))
    # change the metric if you like

    return {'loss': mse_scr, 'status': STATUS_OK}


trials = Trials()

best = fmin(f, params, algo=tpe.suggest, max_evals=8, trials=trials)
print('best:')
print(best)


