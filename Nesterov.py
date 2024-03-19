def f(x): 
  return 10 * x[0]**2 + 0.1 * x[1]**2
def gradf(x):
  return np.array([20 * x[0],0.2 * x[1]])

def gradient_descent(grad, x0, s, n):
    x_k = x0
    fs = [x0]
    for k in range(n):
        x_k1 = x_k - s * grad(x_k)
        fs.append(x_k1)
        x_k = x_k1
    return x_k, np.array(fs)

def nesterov(grad, x0, y0, s, n_iter):
    x_k, y_k = x0, y0
    fs = [f(x0)]
    for k in range(1, n_iter):
        x_k1 = y_k - s * grad(y_k)
        y_k1 = x_k1 + ((k-1)/(k+2)) * (x_k1 - x_k)
        fs.append(f(x_k1))

        y_k = y_k1
        x_k = x_k1
    return x_k, fs

x0 = np.array([.5, .5])
_, fsgd = gradient_descent(gradf,x0, 0.1, 500)
_, fsnd = nesterov(gradf,x0, x0, 0.1, 500)
print(fsgd)
print(fsnd)


