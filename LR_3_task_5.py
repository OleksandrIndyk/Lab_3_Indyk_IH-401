import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
m = 100 #Варіант 9
X = np.linspace(-3, 3, m)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, m)
X = X.reshape(-1, 1) #Перетворення X у формат (m, 1) для сумісності з sklearn
lin_reg = LinearRegression() #Побудова лінійної регресії
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)
poly_features = PolynomialFeatures(degree=3, include_bias=False) #Поліноміальні ознаки (3 ступінь)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression() #Побудова поліноміальної регресії
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
print("X[0]:", X[0]) #Виведення коефіцієнтів
print("X_poly[0]:", X_poly[0])
print("\nЛінійна модель:")
print("intercept:", lin_reg.intercept_)
print("coefficient:", lin_reg.coef_)
print("\nПоліноміальна модель (3 ступінь):")
print("intercept:", poly_reg.intercept_)
print("coefficients:", poly_reg.coef_)
plt.figure(figsize=(10, 6)) #Графік результатів
plt.scatter(X, y, label="Випадкові дані", color="green")
plt.plot(X, y_pred_lin, label="Лінійна регресія", color="red", linewidth=2)
plt.plot(X, y_pred_poly, label="Поліноміальна регресія (3 ступінь)", color="blue", linewidth=2)
plt.title("Порівняння лінійної та поліноміальної регресій (варіант 9)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
