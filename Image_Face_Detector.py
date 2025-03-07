import cv2 # type: ignore
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('Picture.JPG')

#resizing image
width = int(img.shape[1] * 50 / 100)
height = int(img.shape[0] * 50 / 100)
resized_img = cv2.resize(img, (width, height))

#putting the image in gray for better reading by the algorithm
grayscaled_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

#finding the location of the face in the image
#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
face_coordinates = trained_face_data.detectMultiScale(
    grayscaled_img, 
    scaleFactor=1.1,  
    minNeighbors=20,   
    minSize=(30, 30)  
)
#print(face_coordinates)

#drawing the rectangle around the face using the face_coordinates
#parameters: coordinates1, coordinates2 + coordinates1, color of the rectangle, thickness of the rectangle
#ex: cv2.rectangle(img, (47, 57), (155 + 47, 155 + 57), (0, 255, 0), 2)
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

cv2.imshow("LN's Face Detector", resized_img)
cv2.waitKey()

print("Code completed")
