import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hedef fonksiyon
def f(x, y):
    return 0.5 * (x**2 + y**2)

# Gradyan
def grad_f(x, y):
    return np.array([x, y])

# Adafactor parametreleri
beta2 = 0.99
epsilon = 1e-6
learning_rate = 0.1
max_iters = 500

# Başlangıç noktası
x, y = 4.0, 4.0
positions = [(x, y)]

# Başlangıç değerleri
R = np.array([0.0, 0.0])  # Satır ortalamaları
C = np.array([0.0, 0.0])  # Sütun ortalamaları
t = 0  # Iterasyon sayısı

# Adafactor algoritması
for i in range(max_iters):
    t += 1
    grad = grad_f(x, y)
    
    # Satır ortalamalarını güncelle
    R = beta2 * R + (1 - beta2) * grad**2
    
    # Sütun ortalamalarını güncelle
    C = beta2 * C + (1 - beta2) * grad**2
    
    # Kare norm tahmini
    v_hat = R * C / (np.sum(R * C) / (len(R) * len(C)))
    
    # Parametre güncelleme
    x -= learning_rate * grad[0] / (np.sqrt(v_hat[0]) + epsilon)
    y -= learning_rate * grad[1] / (np.sqrt(v_hat[1]) + epsilon)
    
    # Kaydedilen noktalar
    positions.append((x, y))

# Görselleştirme
fig, ax = plt.subplots()
X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = f(X, Y)
ax.contour(X, Y, Z, levels=50, cmap='jet')
point, = ax.plot([], [], 'ro')

def update(i):
    point.set_data([positions[i][0]], [positions[i][1]])  
    return point,

# Animasyonu gif olarak kaydet
ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=50, blit=True)

# Animasyonu gif olarak kaydet
ani.save('adafactor_animation.gif', writer='pillow', fps=40)

print("Animasyon 'adafactor_animation.gif' olarak kaydedildi.")
