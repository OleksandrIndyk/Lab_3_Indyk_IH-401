import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
m = 100 #Варіант 9
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X).ravel() + np.random.uniform(-0.5, 0.5, m)
def plot_learning_curves(model, X, y, title="Криві навчання"): #Функція для побудови кривих навчання
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    sizes = range(1, len(X_train) + 1)
    for m in sizes:
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(sizes, np.sqrt(train_errors), "r-", linewidth=2, label="Навчальна помилка")
    plt.plot(sizes, np.sqrt(val_errors), "b--", linewidth=2, label="Перевірочна помилка")
    plt.xlabel("Кількість навчальних зразків")
    plt.ylabel("Корінь середньоквадратичної помилки (RMSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
lin_reg = LinearRegression() #1. Лінійна модель
plot_learning_curves(lin_reg, X, y, "Криві навчання: Лінійна регресія")
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False) #2. Поліноміальна модель ступеня 2
X_poly_2 = poly_features_2.fit_transform(X)
lin_reg_2 = LinearRegression()
plot_learning_curves(lin_reg_2, X_poly_2, y, "Криві навчання: Поліноміальна регресія (ступінь 2)")
poly_features_10 = PolynomialFeatures(degree=10, include_bias=False) #3. Поліноміальна модель ступеня 10
X_poly_10 = poly_features_10.fit_transform(X)
lin_reg_10 = LinearRegression()
plot_learning_curves(lin_reg_10, X_poly_10, y, "Криві навчання: Поліноміальна регресія (ступінь 10)")
