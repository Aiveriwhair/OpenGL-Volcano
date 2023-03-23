import numpy as np
from scipy.fftpack import fft2, ifft2

class FftWater():
    def phillips(k, max_l):
        """Returns the phillips spectrum for a given wave vector k and a maximum wave length max_l"""
        k_length = np.linalg.norm(k)
        if k_length == 0:
            return 0
        k_length = np.linalg.norm(k)
        k_dot_l = np.dot(k, max_l)
        k_dot_l /= (np.linalg.norm(k) * np.linalg.norm(max_l))
        k_dot_l = np.arccos(k_dot_l)
        k_dot_l = np.cos(k_dot_l)
        k_dot_l = np.power(k_dot_l, 4)
        k_dot_l *= np.exp(-1 / (k_length * k_length))
        k_dot_l /= (k_length * k_length * k_length * k_length)
        return k_dot_l


    def generate_spectrum(N, L, A, wind_dir, seed=0):
        """Generates a spectrum of waves with a given size N, a maximum wave length L, a wind direction wind_dir and a wind speed wind_speed"""
        np.random.seed(seed)
        spectrum = np.zeros((N, N), dtype=np.complex128)
        max_l = np.array([wind_dir[0] * L, wind_dir[1] * L, 0])
        for i in range(N):
            for j in range(N):
                k = np.array([2 * np.pi * i / N, 2 * np.pi * j / N, 0])
                amplitude = np.sqrt(FftWater.phillips(k, max_l)) * A
                spectrum[i, j] = amplitude * FftWater.phillips(k, max_l) * np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
        return spectrum
    
    def generate_heightmap(N, L, A, wind_dir, translateX=0, translateY=0, seed=0):
        spectrum = FftWater.generate_spectrum(N, L, A, wind_dir, seed)
        heightmap = np.real(ifft2(spectrum))
        pad_width = N // 2
        heightmap_padded = np.pad(heightmap, ((pad_width, pad_width), (pad_width, pad_width)), mode='wrap')
        heightmap_padded = np.roll(heightmap_padded, translateX+pad_width, axis=1)
        heightmap_padded = np.roll(heightmap_padded, translateY+pad_width, axis=0)

        # Récupère la hauteur générée à partir de la hauteur paddée
        heightmap = heightmap_padded[pad_width:-pad_width, pad_width:-pad_width]
        return heightmap
