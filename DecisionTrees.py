
# Source
# https://nbviewer.jupyter.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Yacht%20Resistance%20with%20Decision%20Trees%20%26%20Random%20Forests.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Estas lineas ya las hemos visto en LinearRegression, por lo que no las comentare
yacht = pd.read_csv("input/yacht_hydrodynamics.csv", names=["longitudinal_pos", "presmatic_coef", "length_disp",
                                                            "beam-draught_rt", "length-beam_rt", "froude_num",
                                                            "resid_resist"], sep=" ")
yacht = yacht.dropna()
X = yacht.drop(["resid_resist", "longitudinal_pos", "length-beam_rt"], axis=1)
y = yacht["resid_resist"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

# Vamos a utilizar tanto un unico arbol en DecisionTree como varios arboles (Ensemble Learning) en RandomForest
# ________________________________DT___________________________________________________
model = DecisionTreeRegressor()
model.fit(train_scaled, y_train)

print("------------DF-------------------------------")
print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data: ", round(model.score(test_scaled, y_test)*100, 2), "%")
print(model.get_params().keys())
gridParams = {
    'max_depth': np.arange(3, 20), 'min_samples_split': [2, 3, 4, 5]}

grid = GridSearchCV(model, gridParams,
                    verbose=1,
                    cv=5)
grid.fit(train_scaled, y_train)
print("Best params:", grid.best_params_)
print("\nBest score:", grid.best_score_)

params = grid.best_params_

model = DecisionTreeRegressor(**params)
model.fit(train_scaled, y_train)
print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data with best param: ", round(model.score(test_scaled, y_test)*100, 2), "%")
print("MAE: ", mean_absolute_error(y_test, model.predict(X_test)))
# ________________________________RF___________________________________________________
print("------------RF-------------------------------")
model = RandomForestRegressor()
model.fit(train_scaled, y_train)

print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data: ", round(model.score(test_scaled, y_test)*100, 2), "%")

gridParams = {
    'max_depth': np.arange(3, 20), 'min_samples_split': [2, 3, 4, 5], 'n_estimators': np.arange(1, 20)}

grid = GridSearchCV(model, gridParams,
                    verbose=1,
                    cv=5)
grid.fit(train_scaled, y_train)
print("Best params:", grid.best_params_)
print("\nBest score:", grid.best_score_)

params = grid.best_params_

model = RandomForestRegressor(**params)
model.fit(train_scaled, y_train)
print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data with best param: ", round(model.score(test_scaled, y_test)*100, 2), "%")
print("MAE: ", mean_absolute_error(y_test, model.predict(X_test)))


# El best_score con GridSearchCV es peor a la real porque se divide el train dataset en folds
# Tiene mejor pinta esta preicisón :D. Ahora, vamos a analizar algunos parámetros de estos modelos.
# Decission Tree:
#       ~criterion (default: mse)
#               Metrica para calcular la calidad de un split. mse | friedman_mse | mae
#       ~splitter (default: best)
#               Estrategia para dividir los datos en cada nodo. best | random
#               TODO: Poner splitter="random" y nos llevaremos una sorpresa
#       ~max_depth (default: None)
#               Maxima profundidad del arbol
#       ~min_sample_split (default: 2)
#               Minimu numero de ejemplos para dividir un nodo
#       ~max_sample_leaf (default: None)
#               Minimo numero de ejemplos que tiene que haber en un nodo
#       ~min_weight_fraction_leaf (default: 0)
#               Minima fraccion ponderada de la suma total de pesos requeridas en un nodo
#       ~max_features (default: None)
#               Numero de caracteristicas a considerar cuando realizamos el split
#       ~random_state (default: None)
#               Seed usada
#       ~max_leaf_nodes (default: None)
#               Solo crea los mejores nodos posibles
#       ~min_impurity_decrease (default: 0)
#               Un nodo se dividira si crea una impurity mayor o igual al valor indicado
#       ~presort (default: None)
# Random Forest:
#       ~n_estimators (default: 10)
#               Numero de arboles usados durante el ensemble learning
#       ~El resto de parmámetros iguales a DT
