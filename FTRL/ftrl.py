import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fonksiyon
def f(x, y):
    return 0.5 * (x**2 + y**2)

# Gradyan
def grad_f(x, y):
    return np.array([x, y])

# FTRL parametreleri
learning_rate = 1.0  # Öğrenme oranı başlangıcı
l1 = 0.1             # L1 cezası
l2 = 0.1             # L2 cezası
max_iters = 200

# Başlangıç noktası
theta = np.array([4.0, 4.0])

# Birikimli gradyan ve sigma
grad_accum = np.zeros_like(theta)
sigma_accum = np.zeros_like(theta)

# Pozisyonlar (animasyon için)
positions = [theta.copy()]

# FTRL algoritması
for t in range(1, max_iters + 1):
    grad = grad_f(theta[0], theta[1])
    
    # Gradyanları ve sigma'yı güncelle
    grad_accum += grad
    sigma_accum += np.ones_like(theta)  # burada sigma her adımda 1 artıyor
    
    # FTRL güncellemesi
    z = grad_accum
    sigma = sigma_accum
    
    # Proximal adım (soft-thresholding uygulaması)
    theta_new = np.zeros_like(theta)
    for i in range(len(theta)):
        if abs(z[i]) <= l1:
            theta_new[i] = 0.0
        else:
            theta_new[i] = -(1.0 / (sigma[i] + l2)) * (z[i] - l1 * np.sign(z[i]))
    
    # Öğrenme oranını uygula
    theta = learning_rate * theta_new
    
    positions.append(theta.copy())

fig, ax = plt.subplots()
X, Y = np.meshgrid(np.linspace(-5,5,100), np.linspace(-5,5,100))
Z = f(X, Y)
ax.contour(X, Y, Z, levels=50, cmap='jet')
point, = ax.plot([], [], 'ro')

def update_frame(i):
    point.set_data([positions[i][0]], [positions[i][1]])
    return point,

ani = animation.FuncAnimation(fig, update_frame, frames=len(positions), interval=50, blit=True)

# GIF olarak kaydet
ani.save('ftrl_animation.gif', writer='pillow', fps=40)

print("Animasyon başarıyla kaydedildi: ftrl_animation.gif")
