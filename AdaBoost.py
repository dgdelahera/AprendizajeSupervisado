# Source
# https://nbviewer.jupyter.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Yacht%20Resistance%20with%20K%20Nearest%20Neighbors.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Estas lineas ya las hemos visto en LinearRegression, por lo que no las comentare
yacht = pd.read_csv("input/yacht_hydrodynamics.csv", names=["longitudinal_pos", "presmatic_coef", "length_disp",
                                                            "beam-draught_rt", "length-beam_rt", "froude_num",
                                                            "resid_resist"], sep=" ")

# Sustituir NaN con media, mediona o moda es peor que quitar los Nan
yacht = yacht.dropna()
# yacht["presmatic_coef"].fillna((yacht["presmatic_coef"].mean()[0]), inplace=True)
# yacht["presmatic_coef"].fillna((yacht["presmatic_coef"].mode()[0]), inplace=True)
# yacht["presmatic_coef"].fillna((yacht["presmatic_coef"].median()), inplace=True)

# Con esto vemos la correlacion
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(yacht.corr("kendall"))


# Logintudinal_pos tiene una corr con Y de 0.00004 y muy poca con las otras por lo que la descartamos
# Las otras tienen poca corr con Y, por lo que deberiamos meterlas combinandolas entre las que esten relacionadas
# yacht["lengthrt_leghtndisp"] = yacht["length-beam_rt"] / yacht["length_disp"]
X = yacht.drop(["resid_resist", "longitudinal_pos", "length-beam_rt"], axis=1)
y = yacht["resid_resist"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# MinMaxScaler da mejores resultados que StanderScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)


model = AdaBoostRegressor(base_estimator=RandomForestRegressor())
model.fit(train_scaled, y_train)

print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data: ", round(model.score(test_scaled, y_test)*100, 2), "%")
print("Parameters: ", model.get_params())
print("MAE: ", mean_absolute_error(y_test, model.predict(test_scaled)))
# TODO: Se puede mejorar el Grid
gridParams = {
     "n_estimators": [200], 'base_estimator__n_estimators': np.arange(1, 20)}

grid = GridSearchCV(model, gridParams,
                    verbose=1,
                    cv=5)
grid.fit(train_scaled, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)

params = grid.best_params_

model = AdaBoostRegressor(**params)
model.fit(train_scaled, y_train)

print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data with best param: ", round(model.score(test_scaled, y_test)*100, 2), "%")
print("MAE: ", mean_absolute_error(y_test, model.predict(test_scaled)))


# AdaBoostRegressor:
#       ~base_estimator(default: None)
#               Estimador sobre el que se construye el boosted ensemble
#       ~n_estimators(default: 50)
#               Numero de estimadores donde termina el boosteo
#       ~learning_rate (default: 1.)
#               Hace referencia a la contribucción de cada estimador
#       ~loss (default: linear)
#               Error para actualizar los pesos despues de cada iteración linear | sqare | exponential
#       ~random_state (default: None)
#               Seed usada
