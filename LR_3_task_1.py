import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
input_file = "C:\\AI\\data_singlevar_regr.txt" #Завантаження вхідних даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
num_training = int(0.8 * len(X)) #Розбиття даних на навчальний і тестовий набори
num_test = len(X) - num_training
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
regressor = linear_model.LinearRegression() #Створення і навчання моделі
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test) #Прогнозування результатів
plt.scatter(X_test, y_test, color='green', label='Actual data') #Побудова графіка
plt.plot(X_test, y_test_pred, color='black', linewidth=4, label='Regression line')
plt.title('Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
print("Linear regressor performance:") #Оцінка якості моделі
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
output_model_file = 'model.pkl' #Збереження моделі
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
with open(output_model_file, 'rb') as f: #Завантаження моделі та повторне прогнозування
    regressor_model = pickle.load(f)
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
