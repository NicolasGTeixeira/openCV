import cv2

# Determinar o caminho de trabalho basico do programa
workPATH = "/home/nicolas/Documentos/estudos/deteccaoFaces/"

# Determinar o classificador de FACE
classFace = cv2.CascadeClassifier(workPATH + "cascades/haarcascade_frontalface_default.xml")

# Determinar o classificador de OLHOS
classEyes = cv2.CascadeClassifier(workPATH + "cascades/haarcascade_eye.xml")

# Lê a imagem
img = cv2.imread(workPATH + "imagens/pessoas/pessoas4.jpg")

# Converte ela para Cinza
imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Treina a imagem para detectar as faces
facesDetectadas = classFace.detectMultiScale(imgCinza, scaleFactor=1.08, minNeighbors=5)


for (x, y, l, a) in facesDetectadas:
    img = cv2.rectangle(img, (x, y), (x + l, y + a), (0, 0, 0), 2)
    regiao = img[y:y + a, x:x + l]
    regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhosDetectadas = classEyes.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.1, minNeighbors=10)
    for (ox, oy, ol, oa) in olhosDetectadas:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 0), 2)

cv2.imshow("Faces Detectadas", img)
cv2.waitKey()
