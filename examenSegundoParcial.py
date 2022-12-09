import cv2 as cv
import numpy as np
from math import sqrt
from sklearn.cluster import KMeans


def img_kmeans(img, k):
    # Obtener cada canal por separado
    copy = img.copy()
    r = copy[:, :, 0]
    g = copy[:, :, 1]
    b = copy[:, :, 2]

    # Redimensionar cada canal para crear el modelo
    xr = r.reshape((-1, 1))
    xg = g.reshape((-1, 1))
    xb = b.reshape((-1, 1))

    # Concatenar canales redimensionados
    x = np.concatenate((xr, xg, xb), axis=1)

    # Aplicar algoritmo KMeans
    model = KMeans(n_clusters=k)
    model.fit(x)

    clusters = model.cluster_centers_
    labels = model.labels_

    # Asignar valores obtenidos del KMeans a una nueva imagen, comenzando por obtener los
    # resultados por canal y redimensionarlos a las escalas de la imagen original
    rows = xr.shape[0]
    for i in range(rows):
        xr[i] = clusters[labels[i]][0]
        xg[i] = clusters[labels[i]][1]
        xb[i] = clusters[labels[i]][2]

    # Volver a dimensionar los canales a su tamaño original
    xr.shape = r.shape
    xg.shape = g.shape
    xb.shape = b.shape
    xr = xr[:, :, np.newaxis]
    xg = xg[:, :, np.newaxis]
    xb = xb[:, :, np.newaxis]

    # Se concatenan los tres canales obtenidos en una única imagen resultante
    result = np.concatenate((xr, xg, xb), axis=2)
    return (result, clusters.astype(np.uint8))


def umbralize(img, values):
    rows = img.shape[0]
    cols = img.shape[1]
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    result = np.zeros(rows * cols, dtype=np.uint8).reshape(rows, cols)

    # Umbralizar a blanco los valores de la imagen que empaten con los valores
    # del centroide dado
    for i in range(rows):
        for j in range(cols):
            if r[i][j] == values[0] and g[i][j] == values[1] and b[i][j] == values[2]:
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result


def count_pixels(img):
    rows = img.shape[0]
    cols = img.shape[1]

    count = 0
    for i in range(rows):
        for j in range(cols):
            if img[i][j] == 255:
                count += 1
    return count


def get_objects(img, clusters):
    # Crear una imagen umbralizada por cada centroide obtenido
    clusters_array = [umbralize(img, c) for c in clusters]

    # Contar pixeles con intensidad 255 de cada imagen umbralizada
    counts = [count_pixels(c) for c in clusters_array]

    # Crear arreglo de las imagenes umbralizadas junto a su cantidad de
    # pixeles con valor 255
    zipped = list(zip(counts, clusters_array))
    zipped =  sorted(zipped, key=lambda x: x[0])

    return (zipped[1][1], clusters_array)


def half_point(p1, p2):
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    return [x, y]

def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def main():
    # Leer imagen original
    img = cv.imread("./Jit1.JPG")

    # Redimensionar imagen al 15% para poder observarla completa durante el procesamiento
    # de la misma. Esto debido a que al tener una resolución tan grande, mi pantalla es
    # incapaz de proyectarla completamente, permitiéndome ver únicamente unas cuantas
    # rocas de la esquina superior izquierda
    width = int(img.shape[1] * 15 / 100)
    height = int(img.shape[0] * 15 / 100)
    img = cv.resize(img, (width, height))
    img = img.astype(np.uint8)

    # Clonar la imagen original para no alterarla durante el procesamiento
    copy = img.copy()

    # Aplicar el algoritmo KMeans para obtener la imagen segmentada por colores y sus clústers
    # Se pasa la copia de la imagen original y se indican 4 clústers a obtener
    kmeans, clusters = img_kmeans(copy, 4)

    # Se crea una imagen umbralizada por cada clúster obtenido del algoritmo KMeans, obteniendo
    # asi k imágenes umbralizadas dentro del arreglo "all_objects" y la imagen objetivo (la de
    # los jitomates) en la variable "objects"
    objects, all_objects = get_objects(kmeans, clusters)

    # Para eliminar los bordes pequeños resultantes de la umbralización, se suaviza la imagen
    # con los objetos, aplicando el filtro Gaussiano, con un kernel de 5x5 y sigma=0
    gauss = cv.GaussianBlur(objects, (5, 5), 0)

    # Se detectan los bordes de la imagen suavizada utilizando el algoritmo "Canny", el cual
    # recibe un umbral inferior de 50 y uno superior de 150
    canny = cv.Canny(gauss, 50, 150)

    # Usando la función "findContours", se almacenan los bordes en elementos separados, es
    # decir, las coordenadas de cada objeto en la imagen se almacenan en un arreglo
    # diferente dentro de la variable "contours"
    # El parámetro RETR_EXTERNAL obtiene el contorno externo de cada objeto
    # El parámetro CHAIN_APPROX_SIMPLE elimina todos los puntos redundantes de cada contornos
    contours, _ = cv.findContours(canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Imprimir cantidad de objetos encontrados
    print(f"\nSe han encontrado {len(contours)} objetos en la imagen\n")

    # Obtener coordenadas y pintar lineas transversales
    # Usando el siguiente ciclo, se obtiene un rectángulo con área mínima rodeando cada objeto en la imagen,
    # se obtiene el ángulo de inclinación de dicho rectángulo, se obtienen los puntos exactos de la mitad del
    # rectángulo, se obtienen las líneas exactas para posteriormente pintar una línea cortándolo transversalmente
    for i in range(-len(contours), 0):
        if i % 2 == 0:                              # Tomar en cuenta el contorno 0 y 2 (jitomates 2 y 4)
            points = cv.minAreaRect(contours[i])    # Obtener puntos y ántulos del rectángulo
            box = cv.boxPoints(points)
            box = np.int0(box)

            c1, c2, c3, c4 = box        # Aignar cada esquina del rectángulo a una variable individual
            p1 = half_point(c1, c4)     # Se obtiene el punto medio vertical entre dos esquinas
            p2 = half_point(c2, c3)     # Se obtiene el punto medio vertical entre dos esquinas
            d1 = distance(p1, p2)       # Se obtiene la distancia de los dos puntos medios

            # Imprimir coordenadas y distancias
            print(f"Coordenadas del jitomate {-i}:({int(p1[0])}, {int(p1[1])}), ({int(p2[0])}, {int(p2[1])})")
            print(f"Diametro del jitomate {-i}: {d1:.5f}")

            # Pintar lineas obtenidas para cada objeto
            cv.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255,0,0), 2)


    # Ciclo utilizado para imprimir todas las imagenes umbralizadas obtenidas de los clústers
    # resultantes del KMeans
    for i in range(len(all_objects)):
        cv.imshow(f"{i}", all_objects[i])
        cv.imwrite(f"Umbralizada_cluster_{i + 1}.png", all_objects[i])

    # Mostrar todas las imagenes del procesamiento
    cv.imshow("Original", img)                      # Imagen original
    cv.imwrite("Original.png", img)                 # Guardar imagen original
    cv.imshow("Kmeans segmentation", kmeans)        # Imagen segmentada por colores
    cv.imwrite("Kmeans_segmentation.png", kmeans)   # Guardar imagen segmentada por colores
    cv.imshow("Umbralized objects", objects)        # Objetos umbralizados
    cv.imwrite("Umbralized_objects.png", objects)   # Guardar imagen objetos umbralizados
    cv.imshow("Gauss smoothed", gauss)              # Suavizado de objetos umbralizados
    cv.imwrite("Gauss_smoothed.png", gauss)         # Guardar suavizado de objetos umbralizados
    cv.imshow("Bordered objects", canny)            # Contornos de objetos umbralizados
    cv.imshow("Bordered_objects.png", canny)        # Guardar contornos de objetos umbralizados
    cv.waitKey()


if __name__ == "__main__":
    main()
