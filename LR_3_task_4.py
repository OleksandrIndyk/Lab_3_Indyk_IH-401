import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
diabetes = datasets.load_diabetes() #Завантаження набору даних про діабет
X = diabetes.data
y = diabetes.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0) #Розбиття на тренувальні та тестові дані
regr = linear_model.LinearRegression() #Створення та навчання моделі лінійної регресії
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest) # Прогнозування
print("Коефіцієнти моделі:", regr.coef_) #Виведення коефіцієнтів
print("intercept:", round(regr.intercept_, 2))
print("R2 score =", round(r2_score(ytest, ypred), 2)) #Метрики якості моделі
print("Mean absolute error =", round(mean_absolute_error(ytest, ypred), 2))
print("Mean squared error =", round(mean_squared_error(ytest, ypred), 2))
fig, ax = plt.subplots() #Побудова графіку "Передбачено vs Виміряно"
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), color='green')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.set_title('Залежність між виміряними та передбаченими значеннями')
plt.grid(True)
plt.show()
