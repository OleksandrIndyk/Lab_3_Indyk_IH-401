import numpy as np
import matplotlib.pyplot as plt
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7]) #Заповнення векторів x і y
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])
coeffs = np.polyfit(x, y, deg=4) #Отримання коефіцієнтів полінома 4-го степеня
print("Коефіцієнти полінома:", coeffs)
p = np.poly1d(coeffs) #Визначення полінома як функції
x_vals = np.linspace(0.05, 0.75, 300) #Побудова графіка
y_vals = p(x_vals)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="Інтерполяційний поліном", color='blue')
plt.scatter(x, y, color='red', label="Вихідні точки", zorder=5)
plt.title("Інтерполяція поліномом 4-го степеня")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
x_test = [0.2, 0.5] #Обчислення значень функції в проміжних точках
y_test = p(x_test)
print(f"Значення функції у точці x = 0.2: {y_test[0]:.4f}")
print(f"Значення функції у точці x = 0.5: {y_test[1]:.4f}")
