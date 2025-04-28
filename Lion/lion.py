import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Küçük bir 2D loss fonksiyonu
def loss_fn(theta):
    return theta[0]**2 + theta[1]**2

def grad_fn(theta):
    return 2 * theta

# Ayarlar
beta1 = 0.9
eta = 0.1
n_steps = 50

theta = np.array([2.0, 2.5])
momentum = np.zeros_like(theta)

trajectory = [theta.copy()]

for step in range(n_steps):
    grad = grad_fn(theta)
    momentum = beta1 * momentum + (1 - beta1) * grad
    theta = theta - eta * np.sign(momentum)
    trajectory.append(theta.copy())

trajectory = np.array(trajectory)

fig, ax = plt.subplots(figsize=(6,6))
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

ax.contourf(X, Y, Z, levels=50, cmap='coolwarm')
point, = ax.plot([], [], 'bo', markersize=8)
path, = ax.plot([], [], 'b-', lw=1)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Lion Optimizer ile Parametre Güncellemesi")

def init():
    point.set_data([], [])
    path.set_data([], [])
    return point, path

def update(frame):
    point.set_data([trajectory[frame][0]], [trajectory[frame][1]])  
    path.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    return point, path

ani = animation.FuncAnimation(fig, update, frames=len(trajectory), init_func=init, interval=50, blit=True)

# Eğer kaydetmek istiyorsan:
ani.save('lion_optimizer.gif', writer='pillow', fps=40)

plt.show()
