import numpy as np
import cv2 as cv

#carrega xml de treinamento de identificação de faces
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#carrega imagens
img = cv.imread('candidatos.jpg')
#converte para cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#detecta faces: minNeighbors mínimo de visinhos
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#detecta faces: scale Factor é a escala da imagem, tamanho mínimpo
#faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minSize= (50,50))

#tamanho mínimo padrão
#faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3 )

# multi escala
#faces = face_cascade.detectMultiScale(gray)

#Desenha retangulos pelo x e y do ponto com largura e altura
for (x,y,w,h) in faces:
    #desenha cada um dos retangulos
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
