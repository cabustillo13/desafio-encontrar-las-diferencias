
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen a analizar
imagen = cv2.imread("./Imagenes recortadas/prueba1.jpg")
imagenc = imagen.copy()

# Imagen de referencia o patron a comparar
referencia = cv2.imread("./Imagenes recortadas/prueba2.jpg")
refc = referencia.copy()

# Convertir las imagenes a escala de grises
imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
referenciaGris = cv2.cvtColor(referencia, cv2.COLOR_BGR2GRAY)

# Calcular el indice de similitud estructural (SSIM) entre las dos imagenes
# Devuelve la imagen con las diferencias encontradas
(score, diff) = compare_ssim(imagenGris, referenciaGris, full=True)
diff = (diff * 255).astype("uint8")

# Devuelve la similitud estructural determinada
print("SSIM: {}".format(score))

# Resaltar los contornos y regiones donde difieren ambas imagenes
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Aplicamos un loop para cada contorno detectado
for c in cnts:
	# Calcular el cuadro delimitador
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(referencia, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
# Mostrar resultados
f, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(refc)
axes[0].set_title('Referencia')
axes[1].imshow(imagenc)
axes[1].set_title('Imagen')
axes[2].imshow(diff)
axes[2].set_title('Diferencias')

for ax in axes:
    ax.axis('off')
plt.show()
