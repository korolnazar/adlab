import numpy as np
import matplotlib.pyplot as plt

def generate_data(n=100, k=2, b=5, noise=5):
    x = np.linspace(0, 10, n)
    y = k * x + b + np.random.randn(n) * noise
    return x, y

def least_squares(x, y):
    n = len(x)
    x_mean, y_mean = np.mean(x), np.mean(y)
    k = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - k * x_mean
    return k, b

def gradient_descent(x, y, lr=0.01, n_iter=1000):
    k, b = np.random.randn(), np.random.randn()
    n = len(x)
    errors = []
    for _ in range(n_iter):
        y_pred = k * x + b
        error = np.mean((y - y_pred) ** 2)
        errors.append(error)
        k -= lr * (-2 / n) * np.sum(x * (y - y_pred))
        b -= lr * (-2 / n) * np.sum(y - y_pred)
    return k, b, errors

x, y = generate_data()
k_ls, b_ls = least_squares(x, y)
k_gd, b_gd, errors = gradient_descent(x, y)
k_np, b_np = np.polyfit(x, y, 1)

plt.scatter(x, y, label='Data')
plt.plot(x, k_ls * x + b_ls, label='Least Squares', color='r')
plt.plot(x, k_gd * x + b_gd, label='Gradient Descent', color='g')
plt.plot(x, k_np * x + b_np, label='Numpy polyfit', color='b', linestyle='dashed')
plt.legend()
plt.show()

plt.plot(errors)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Error reduction over iterations')
plt.show()

print(f'Least Squares: k={k_ls:.2f}, b={b_ls:.2f}')
print(f'Gradient Descent: k={k_gd:.2f}, b={b_gd:.2f}')
print(f'Numpy polyfit: k={k_np:.2f}, b={b_np:.2f}')
