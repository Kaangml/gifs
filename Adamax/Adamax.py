import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hedef fonksiyon
def f(x, y):
    return 0.5 * (x**2 + y**2)

# Gradyan
def grad_f(x, y):
    return np.array([x, y])

# Adamax parametreleri
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-6
learning_rate = 0.1
max_iters = 500

# Başlangıç noktası
x, y = 4.0, 4.0
positions = [(x, y)]

# Başlangıç değerleri
m = np.array([0.0, 0.0])  # İlk moment
u = np.array([0.0, 0.0])  # Sonsuz norm
t = 0  # Iterasyon sayısı

# Adamax algoritması
for i in range(max_iters):
    t += 1
    grad = grad_f(x, y)
    
    # İlk moment güncellemesi (ortalama gradyan)
    m = beta1 * m + (1 - beta1) * grad
    
    # Sonsuz norm güncellemesi
    u = np.maximum(beta2 * u, np.abs(grad))
    
    # Büyüklük düzeltmesi
    m_hat = m / (1 - beta1**t)
    
    # Parametre güncelleme
    x -= learning_rate * m_hat[0] / (u[0] + epsilon)
    y -= learning_rate * m_hat[1] / (u[1] + epsilon)
    
    # Kaydedilen noktalar
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
ani.save('adamax_animation.gif', writer='pillow', fps=40)

print("Animasyon 'adamax_animation.gif' olarak kaydedildi.")
