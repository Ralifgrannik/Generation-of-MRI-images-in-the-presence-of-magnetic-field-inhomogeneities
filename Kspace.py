import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Настройка и исходные данные (как в оригинальном скрипте) ---

# Загрузка и нормализация (замените на ваш путь, если нужно)
try:
    img = cv2.imread("/content/Testing/notumor/Te-no_0060.jpg", 0)
    img = cv2.resize(img, (256, 256))
except:
    # Запасной вариант: создаем тестовое изображение
    print("Используется тестовое синусоидальное изображение.")
    H, W = 256, 256
    x = np.linspace(0, 8*np.pi, W)
    y = np.linspace(0, 8*np.pi, H)
    X, Y = np.meshgrid(x, y)
    img = np.sin(X) * np.cos(Y)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)

img_norm = img.astype(float) / 255.0

# 1. Прямое преобразование Фурье (для получения k-пространства)
k_space_centered = np.fft.fftshift(np.fft.fft2(img_norm))

# -------------------------------------------------------------
# 2. ФУНКЦИЯ ПОШАГОВОЙ РЕКОНСТРУКЦИИ
# -------------------------------------------------------------

def visualize_reconstruction_steps(k_centered):
    """
    Визуализирует три ключевых этапа обратного преобразования Фурье.

    k_centered: k-пространство с нулевой частотой в центре.
    """
    
    # ------------------------------------------------
    # ЭТАП 1: Исходное k-пространство (Центрированное)
    # ------------------------------------------------
    
    # Визуализация амплитуды k-пространства
    k_amp_log = np.log1p(np.abs(k_centered))
    
    # ------------------------------------------------
    # ЭТАП 2: Обратный сдвиг нулевой частоты (iFFTshift)
    # ------------------------------------------------
    
    # Сдвиг нулевой частоты из центра в угол для корректного iFFT
    k_shifted_for_ifft = np.fft.ifftshift(k_centered)
    
    # Визуализация амплитуды k-пространства после сдвига
    k_shifted_amp_log = np.log1p(np.abs(k_shifted_for_ifft))

    # ------------------------------------------------
    # ЭТАП 3: Обратное преобразование Фурье (iFFT2)
    # ------------------------------------------------
    
    # Выполнение iFFT. Результат - комплексное изображение
    complex_image = np.fft.ifft2(k_shifted_for_ifft)
    
    # ------------------------------------------------
    # ЭТАП 4: Взятие Амплитуды (Модуля)
    # ------------------------------------------------
    
    # Реконструкция - берем модуль комплексного изображения
    reconstructed_image = np.abs(complex_image)
    
    # ------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------
    
    plt.figure(figsize=(15, 12))
    plt.suptitle("Пошаговая визуализация реконструкции (Обратное преобразование Фурье)", fontsize=16)

    # 1. k-пространство (Центрированное)
    plt.subplot(2, 3, 1)
    plt.imshow(k_amp_log, cmap='gray')
    plt.title("1. k-пространство (log|k|)\n(Нулевая частота в центре)", color='blue')
    plt.axis('off')
    
    # 2. k-пространство (Сдвинутое)
    plt.subplot(2, 3, 2)
    plt.imshow(k_shifted_amp_log, cmap='gray')
    plt.title("2. k-пространство после iFFTshift\n(Нулевая частота в углу)", color='red')
    plt.axis('off')

    # 3. Комплексное изображение (Реальная часть)
    plt.subplot(2, 3, 4)
    plt.imshow(np.real(complex_image), cmap='gray')
    plt.title("3. Реальная часть iFFT(k_shifted)", color='green')
    plt.axis('off')
    
    # 4. Комплексное изображение (Мнимая часть)
    plt.subplot(2, 3, 5)
    plt.imshow(np.imag(complex_image), cmap='gray')
    plt.title("4. Мнимая часть iFFT(k_shifted)", color='green')
    plt.axis('off')

    # 5. Реконструированное изображение (Модуль)
    plt.subplot(2, 3, 6)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("5. Окончательная реконструкция: Амплитуда (Модуль)", color='purple')
    plt.axis('off')
    
    # 6. Исходное изображение для сравнения
    plt.subplot(2, 3, 3)
    plt.imshow(img_norm, cmap='gray')
    plt.title("Исходное изображение (для сравнения)")
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Запуск визуализации
visualize_reconstruction_steps(k_space_centered)