## Source
## https://nbviewer.jupyter.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Yacht%20Resistance%20with%20K%20Nearest%20Neighbors.ipynb



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Estas lineas ya las hemos visto en LinearRegression, por lo que no las comentare
yacht = pd.read_csv("input/yacht_hydrodynamics.csv", names=["longitudinal_pos", "presmatic_coef", "length_disp", "beam-draught_rt",
                                                           "length-beam_rt", "froude_num", "resid_resist"], sep=" ")
yacht = yacht.dropna()
X = yacht.drop(["resid_resist"], axis=1)
y = yacht["resid_resist"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

model = KNeighborsRegressor()
model.fit(train_scaled, y_train)
print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data: ", round(model.score(test_scaled, y_test)*100, 2), "%")
print("Best params: ", model.get_params())

gridParams = {
    'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15], 'leaf_size': [5, 15, 20, 30, 40],
    'p': [1, 2]}

grid = GridSearchCV(model, gridParams,
                    verbose=1,
                    cv=5)
grid.fit(train_scaled, y_train)
print("Best params:", grid.best_params_)
print("\nBest score:", grid.best_score_)

params = grid.best_params_

model = KNeighborsRegressor(**params)
model.fit(train_scaled, y_train)
print("Accuracy on train data: ", round(model.score(train_scaled, y_train)*100, 2), "%")
print("Accuracy on test data with best param: ", round(model.score(test_scaled, y_test)*100, 2), "%")

# Mejora a LR pero no mejora a DT
# Vamos a ver algunos de los parámetros
#       ~n_neighbors(default: 5)
#               Numero de vecinos que se selecciona
#       ~weights (default: uniform)
#               Funcion usada en predicción
#       ~algorithm (default: auto)
#               Algoritmo usado para computar los vecinos
#       ~leaf_size (default: 30)
#               Tamaño de ramificación que se le pasa a BallTree o KDTree
#       ~p (default: 2)
#               Parametro de Minkowski metric. 1=Distancia Manhattan, 2 Distancia Euclidean
#       ~metric (default: minkowski)
#               Metria usada para el arbol



