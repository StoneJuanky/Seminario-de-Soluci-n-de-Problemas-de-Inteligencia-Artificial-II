import numpy as np
import matplotlib.pyplot as plt

def F(x1, x2):
 return 10- np.exp(-((x1**2) + (x2**2)))

def G(x, y):
 return np.array([(1- 2*(x**2))*np.exp(-x**2-y**2),-2*x*y*np.exp(-x**2-y**2)])

def gradient_descent(lr, max_iterations):
 x1 = np.random.uniform(-1, 1)
 x2 = np.random.uniform(-1, 1)
 error1 = []
 for i in range(max_iterations):
 grad = G(x1, x2)
 x1-= lr * grad[0]
 x2-= lr * grad[1]
 x1 = max(-1, min(x1, 1))
 x2 = max(-1, min(x2, 1))
 error = F(x1, x2)
 error1.append(error)
 if i % 100 == 0:
     print(f"Iteracion numero: {i}: F({x1}, {x2}) = {error}")
 return x1, x2, errors
 
lr = 0.01
max_iterations = 1000

opt_x1, opt_x2, errors = gradient_descent(lr, max_iterations)

plt.plot(range(len(errors)), errors)
plt.xlabel('Numero de iteraciones')
plt.ylabel('Error')
plt.title('Convergencia del error durante el descenso del gradiente')
plt.show()

print("\n\n\n")
print(f"Optimizacion evaluada para: x1 = {opt_x1}, x2 = {opt_x2}")
print(f"Optimizacion para la funcion F(x1, x2) = {F(opt_x1, opt_x2)}")