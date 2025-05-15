import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
x = np.array([0.3, 1, 1.5, 2.2, 3.6, 4.5]).reshape(-1, 1) #Дані
y = np.array([5, 10, 13, 16, 17, 18])
model = LinearRegression() #Лінійна регресія
model.fit(x, y)
y_pred = model.predict(x)
a = model.intercept_ #Параметри моделі
b = model.coef_[0]
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"Рівняння регресії: y = {a:.2f} + {b:.2f}x") #Вивід параметрів
print(f"Коефіцієнт детермінації R² = {r2:.4f}")
print(f"Середньоквадратична помилка (MSE) = {mse:.2f}")
plt.figure(figsize=(8, 5)) #Графік
plt.scatter(x, y, color='red', label='Експериментальні точки')
plt.plot(x, y_pred, color='green', label='Лінійна апроксимація')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Лінійна регресія (метод найменших квадратів)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
