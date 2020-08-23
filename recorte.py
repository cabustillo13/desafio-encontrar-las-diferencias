import cv2

#Funcion para recortar imagenes
def recortar(img, x, y, w, h):
    recortarImagen = img[y:y+h, x:x+w]
    return recortarImagen

#Ingresar imagen a analizar
numero = input("Ingrese numero de la imagen: ")
imagen = cv2.imread("./Imagenes/prueba"+str(numero)+".jpg")

#Determinar dimensiones de la imagen para recortarla en dos imagenes
alto,ancho,canales = imagen.shape

#Esta imagen tiene una separacion entre ambos valores y varia para cada imagen de entrada
#No se consideraron si la imagen presenta bordes
separacion = 4 

imagen1 = recortar(imagen,0,0,ancho/2-separacion,alto)
imagen2 = recortar(imagen,ancho/2+separacion,0,ancho,alto)
#Guardar imagenes recortadas
cv2.imwrite("./Imagenes recortadas/prueba1.jpg",imagen1)
cv2.imwrite("./Imagenes recortadas/prueba2.jpg",imagen2)
