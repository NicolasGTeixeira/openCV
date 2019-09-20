import cv2

video = cv2.VideoCapture(0)

# Declarando o lugar de trabalho
workPath = "/l/disk0/nicolas/Documentos/pessoal/OpenCV/"

# Declarando o Classificador
classificador = cv2.CascadeClassifier(workPath + "cascades/haarcascade_frontalface_default.xml")

# Determinar o classificador de OLHOS
classEyes = cv2.CascadeClassifier(workPath + "cascades/haarcascade_eye.xml")

while True:
    conectado, frame = video.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificador.detectMultiScale(frameCinza, minSize=(80,80))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (153,204,50), 2)

        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectadas = classEyes.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.1, minNeighbors=10)
        for (ox, oy, ol, oa) in olhosDetectadas:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 0), 2)

        cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

print("Faces Destectadas", len(frame))
video.release()
cv2.destroyAllWindows()