# Eski import'u kaldır
# from data import get_mnist

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path

def get_mnist():
    with np.load(f"{Path(__file__).parent/'data/mnist.npz'}") as f:
        x, y = f["x_train"], f["y_train"]
    return (x.astype("float32").reshape(x.shape[0], -1) / 255,
            np.eye(10)[y])

# Matplotlib'i interaktif moda al
plt.ion()

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3

# Eğitim döngüsü
for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    nr_correct = 0

# Matplotlib'i interaktif moddan çık
plt.ioff()

class DrawingCanvas:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.drawing = np.zeros((28, 28))
        self.ax1.set_title("Çizim Alanı")
        
        # Çizim alanı ayarları
        self.image = self.ax1.imshow(self.drawing, cmap='Greys', vmin=0, vmax=1)
        self.ax1.grid(True, color='gray', alpha=0.3)
        self.ax1.set_xticks(np.arange(-.5, 28, 1), [])
        self.ax1.set_yticks(np.arange(-.5, 28, 1), [])
        
        # Mouse event bağlantıları
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.last_x = None
        self.last_y = None
        self.is_drawing = False
        
        # Butonlar
        self.button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.button = Button(self.button_ax, 'Tahmin Et')
        self.button.on_clicked(self.predict)
        
        self.clear_ax = plt.axes([0.65, 0.05, 0.1, 0.075])
        self.clear_button = Button(self.clear_ax, 'Temizle')
        self.clear_button.on_clicked(self.clear)
        
        self.ax2.set_title("Tahmin")
        self.ax2.axis('off')

    def draw_line(self, x1, y1, x2, y2):
        """İki nokta arasına çizgi çizer"""
        if x1 is None or x2 is None:
            return
        
        # Çizgi noktalarını hesapla
        points = np.linspace([x1, y1], [x2, y2], num=20)
        
        brush_size = 1.5  # Çizgi kalınlığı
        for px, py in points:
            for i in range(max(0, int(py-brush_size)), min(28, int(py+brush_size+1))):
                for j in range(max(0, int(px-brush_size)), min(28, int(px+brush_size+1))):
                    distance = np.sqrt((i-py)**2 + (j-px)**2)
                    if distance <= brush_size:
                        intensity = 1.0 - (distance / brush_size) * 0.3
                        self.drawing[i, j] = min(1.0, self.drawing[i, j] + intensity)

    def draw_pixel(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            if 0 <= x < 28 and 0 <= y < 28:
                # Önceki noktayla şimdiki nokta arasına çizgi çiz
                if self.last_x is not None:
                    self.draw_line(self.last_x, self.last_y, x, y)
                
                self.last_x, self.last_y = x, y
                self.image.set_data(self.drawing)
                self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes == self.ax1:
            self.is_drawing = True
            self.last_x = None
            self.last_y = None
            self.draw_pixel(event)

    def on_motion(self, event):
        if self.is_drawing and event.inaxes == self.ax1:
            self.draw_pixel(event)

    def on_release(self, event):
        self.is_drawing = False
        self.last_x = None
        self.last_y = None

    def predict(self, event):
        # Görüntüyü işle
        processed_img = self.preprocess_image()
        img = processed_img.reshape(784, 1)
        
        # Tahmin yap
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        
        # Tahmin sonuçlarını göster
        probabilities = 1 / (1 + np.exp(-o))
        predicted = o.argmax()
        
        self.ax2.clear()
        self.ax2.text(0.5, 0.5, f'Tahmin: {predicted}\nGüven: {probabilities[predicted][0]:.2f}', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20)
        self.ax2.axis('off')
        self.fig.canvas.draw_idle()

    def preprocess_image(self):
        """Çizimi MNIST formatına uygun hale getirir"""
        # Görüntüyü kopyala
        img = self.drawing.copy()
        
        # Görüntüyü merkeze taşı
        coords = np.where(img > 0)
        if len(coords[0]) > 0:  # Eğer çizim varsa
            top, bottom = coords[0].min(), coords[0].max()
            left, right = coords[1].min(), coords[1].max()
            
            # Merkezi hesapla
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            
            # Ne kadar kaydıracağımızı hesapla
            shift_y = 14 - center_y
            shift_x = 14 - center_x
            
            # Görüntüyü kaydır
            if shift_y != 0 or shift_x != 0:
                temp = np.zeros_like(img)
                for i in range(28):
                    for j in range(28):
                        new_i = i + shift_y
                        new_j = j + shift_x
                        if 0 <= new_i < 28 and 0 <= new_j < 28:
                            temp[new_i, new_j] = img[i, j]
                img = temp
        
        return img

    def clear(self, event):
        self.drawing = np.zeros((28, 28))
        self.image.set_data(self.drawing)
        self.ax2.clear()
        self.ax2.axis('off')
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    canvas = DrawingCanvas()
    plt.show()
