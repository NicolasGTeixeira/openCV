import cv2

# Determinar o caminho de trabalho basico do programa
workPATH = "/home/nicolas/Documentos/estudos/deteccaoFaces/"

# Determinar o classificador de OLHOS
classEyes = cv2.CascadeClassifier(workPATH + "cascades/haarcascade_eye.xml")

# Lê a imagem
img = cv2.imread(workPATH + "imagens/pessoas/beatles.jpg")

# Converte ela para Cinza
imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Treina a imagem para detectar as faces
olhosDetectadas = classEyes.detectMultiScale(imgCinza, scaleFactor=1.07)


for (x, y, l, a) in olhosDetectadas:
    img = cv2.rectangle(img, (x, y), (x + l, y + a), (0, 0, 0), 2)

cv2.imshow("Olhos Detectadas", img)
cv2.waitKey()
