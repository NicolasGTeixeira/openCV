import cv2

# Determinar o caminho de trabalho basico do programa
workPATH = "/home/nicolas/Documentos/estudos/deteccaoFaces/"

# Determina o classificador que será usado
classificadorGato = cv2.CascadeClassifier(workPATH + "cascades/haarcascade_frontalcatface.xml")

# Input da imagem a ser classificada
img = cv2.imread(workPATH + "imagens/gato/gato3.jpg")

# Convertendo a imagem para a escala de cinza
imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Classificando a imagem
catsDetects = classificadorGato.detectMultiScale(imgCinza, scaleFactor=1.02)

# Desenha o retângulo
for (x, y, l, a) in catsDetects:
    cv2.rectangle(img, (x, y), (x + l, y + a), (255, 0, 255), 2)

cv2.imshow("Gatos Detecctados", img)
cv2.waitKey()