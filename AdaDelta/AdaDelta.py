import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hedef fonksiyon
def f(x, y):
    return 0.5 * (x**2 + y**2)

# Gradyan
def grad_f(x, y):
    return np.array([x, y])

# Adadelta parametreleri
rho = 0.9
epsilon = 1e-6
max_iters = 500

# Başlangıç noktası
x, y = 4.0, 4.0
positions = [(x, y)]

# Başlangıç değerleri
Eg2 = np.array([0.0, 0.0])
Edx2 = np.array([0.0, 0.0])

# Adadelta algoritması
for i in range(max_iters):
    grad = grad_f(x, y)
    Eg2 = rho * Eg2 + (1 - rho) * grad**2
    dx = - (np.sqrt(Edx2 + epsilon) / np.sqrt(Eg2 + epsilon)) * grad
    Edx2 = rho * Edx2 + (1 - rho) * dx**2
    x += dx[0]
    y += dx[1]
    positions.append((x, y))

fig, ax = plt.subplots()
X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = f(X, Y)
ax.contour(X, Y, Z, levels=50, cmap='jet')
point, = ax.plot([], [], 'ro')

def update(i):
    point.set_data([positions[i][0]], [positions[i][1]])  
    return point,

ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=50, blit=True)

# Animasyonu gif olarak kaydet
ani.save('adadelta_animation.gif', writer='pillow', fps=40)

print("Animasyon 'adadelta_animation2.gif' olarak kaydedildi.")
