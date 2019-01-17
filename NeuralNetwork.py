# Source:
# https://www.simonwenkel.com/2018/09/08/revisiting-ml-datasets-yacht-hydrodynamics.html

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Estas lineas ya las hemos visto en LinearRegression, por lo que no las comentare
yacht = pd.read_csv("input/yacht_hydrodynamics.csv", names=["longitudinal_pos", "presmatic_coef", "length_disp",
                                                            "beam-draught_rt", "length-beam_rt", "froude_num",
                                                            "resid_resist"], sep=" ")

# Sustituir NaN con media, mediona o moda es peor que quitar los Nan
yacht = yacht.dropna()


# Logintudinal_pos tiene una corr con Y de 0.00004 y muy poca con las otras por lo que la descartamos
# Las otras tienen poca corr con Y, por lo que deberiamos meterlas combinandolas entre las que esten relacionadas
# yacht["lengthrt_leghtndisp"] = yacht["length-beam_rt"] / yacht["length_disp"]
X = yacht.drop(["resid_resist"], axis=1)
y = yacht["resid_resist"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)


model = Sequential()
model.add(Dense(train_scaled.shape[1], input_dim=train_scaled.shape[1], activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

earlystopping = EarlyStopping(monitor='val_mean_absolute_error',
                              patience=5,
                              verbose=1, mode='auto')

model.fit(train_scaled, y_train, batch_size=128, epochs=1000, validation_data=(test_scaled, y_test), callbacks=[earlystopping])


print("MAE: ", round(model.evaluate(test_scaled, y_test, batch_size=128)[1], 2))
