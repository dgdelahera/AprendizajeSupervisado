


Model | Accuracy | mae | Parameters
--- | --- | --- | ---
Linear Regression | 45.28% | 6.10 | Default
k-Neighbors | 95.88 % | 1.23 | Leaf_Size: 40, N_Neighbors: 2
Decision Trees | 99.29 % | 0.76 | Max_Depth: None 
Random Forest |  99.36 %| 0.42 | Max_Depth: None, N_Estimators:10
Neural Network | - | 0.57 | Dense, Dense(9), Dense(1), ReLu, adam, Epochs: 2000
AdaBoost + RF| 99.22 % | 0.46 | N_Estimators: 10 , RF_Estimators: 10
XGBoost |**99.73 %** | **0.24** | N_Estimators: 899

The GridSearchCV is not working properly because the dataset is too small for a correct cross validation, so the params found are no the optimal.
