
## Source:
## https://nbviewer.jupyter.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Yacht%20Resistance%20with%20Linear%20Regression.ipynb

import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# Leemos el dataset pasandole el nombre de las cloumnas
yacht = pd.read_csv("input/yacht_hydrodynamics.csv", names=["longitudinal_pos", "presmatic_coef", "length_disp", "beam-draught_rt",
                                                           "length-beam_rt", "froude_num", "resid_resist"], sep=" ")

# Estos print los vamos descomentando para ir lo que va saliendo, pero se comentan para no tener demasiada info en la consola
# print(yacht.head())
# print(yacht["longitudinal_pos"])


# Analizamos si hay algún valor incompleto
# print(yacht.isnull().values.any())

# Vamos a encontrar esos valores incompletos
#print(yacht.describe())

# Todas las columnas tienen 308 valores menos "presmatic_coef"
# En este caso, vamos a eliminar dichos valores, aunque se podrian rellenar con la moda o la media
# TODO: Probar precision si rellenamos los valores con moda o media
yacht = yacht.dropna()

# En X guardaremos todas las variables dependientes. En y la variable independiente/objetivo
X = yacht.drop(["resid_resist"], axis=1)
y = yacht["resid_resist"]

# Separamos el dataset en train 80% y test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#print(X_train.shape)
#print(X_test.shape)


# Todas las variables son numéricas. Tenemos dos opciones, o agruparlas o estandarizarlas. En este caso vamos a estandarizarlas
# entre 0 y 1.
# TODO: Probar a agrupar algunas de las variables. Para esto deberemos tener un alto conocimiento del dataset y crear grupos con significado
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

# El modelo elegido es LinearRegression. Para llegar a esta elección tendriamos que haber analizado previamente los datos. Este modelo
# funciona cuadno la relacion entre las variales es lineal
# TODO: Probar hiperparametros
model = LinearRegression()
model.fit(train_scaled, y_train)

# Calculamos los errores en el dataset de test.
y_test_pred = model.predict(test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test,y_test_pred)
print("mse = ",test_mse," & mae = ",test_mae," & rmse = ", sqrt(test_mse))

# Para tener una idea mejor del funcionamiento del modelo, vamos a calcular la precision
print("Accuracy on test data: ", round(model.score(test_scaled, y_test)*100,2),"%")

# La metrica de referrencia va a ser mae. En este caso es 6.109. Vamos a ver como podemos mejorarlo
# LinearRegressor tiene unos parametros que podemos cambiar para minimizar el error
#       ~fit_intercept (default: True)
#               Calcula el intercept del modelo. Es util para cuando los datos no estan centrados
#       ~normalize (default: False)
#               Normaliza quitando la media y haciendo una norm L2. Si queremos usar fit_intercept sin normalizar
#               tendremos que normalizarlo por nuesta cuenta
#       ~n_jobs (default: 1)
#               Numero de trabajos para realizar la computacion. Con -1 todas se usan.
#


