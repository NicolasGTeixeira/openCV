import cv2

# Declarando o lugar de trabalho
workPath = "/home/nicolas/Documentos/estudos/deteccaoFaces/"

# Declarando o Classificador
classificador = cv2.CascadeClassifier(workPath + "cascades/haarcascade_frontalface_default.xml")

# Leitura da imagem
imagem = cv2.imread(workPath + "pessoas/pessoas2.jpg")

# Printando imagem sem modificação
cv2.imshow("Original", imagem)

# Converteno em cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Roda o modelo e detecta as faces
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))

# Printa quantas faces foram
print(len(facesDetectadas))

# Printa as informações de onde ele achou o rosto
print(facesDetectadas)

for (x, y, l, a) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 255), 2)

cv2.imshow("Faces Encontradas", imagem)
cv2.waitKey()