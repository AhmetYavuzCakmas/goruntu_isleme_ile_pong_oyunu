import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#Tüm görüntüleri içe aktarma
imgBackground = cv2.imread("Resources/Background.png")
imgBackground = cv2.resize(imgBackground, (1280,720))
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png",cv2.IMREAD_UNCHANGED)
imgBall = cv2.resize(imgBall, (100,100))
imgBat1 = cv2.imread("Resources/bat1.png",cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png",cv2.IMREAD_UNCHANGED)

#HEL Dedektörü
detector = HandDetector(detectionCon=0.8, maxHands=2)

#Değişkenler
ballPos = [100,100]
speedX = 15
speedY = 15
gameOver = False 
score = [0,0]

print("Ball:", imgBall.shape)
print("Bat1:", imgBat1.shape)
print("Bat2:", imgBat2.shape)
while True:
    _, img = cap.read()
    img = cv2.flip(img,1)
    imgRaw = img.copy()

    # eli ve yer işaretlerini bul
    hands,img = detector.findHands(img,flipType=False) #with draw

    # arka planı kameraya uydur
    imgBackground = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))

    #arkaplan görüntüsünün kaplanması
    img = cv2.addWeighted(img,0.2,imgBackground,0.8,0)

    #Elleri kontrol et
    if hands:
        for hand in hands:
            x,y,w,h = hand['bbox']
            h1,w1, _ = imgBat1.shape
            y1 = y - h1 //2
            y1 = np.clip(y1,20,415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img,imgBat1,(59,y1))
                if 59<ballPos[0] <59 + w1 and y1 < ballPos[1] < y1+h1:
                    speedX = -speedX
                    speedX += 1 if speedX > 0 else -1
                    ballPos[0] +=30
                    score[0] +=1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img,imgBat2,(1195,y1))
                if 1195-w1<ballPos[0] <1195  and y1 < ballPos[1] < y1+h1:
                    speedX = -speedX
                    speedX += 1 if speedX > 0 else -1
                    ballPos[0] -=30
                    score[1] +=1

    #oyun bitti 
    if ballPos[0] < 40 or ballPos[0] >1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585,360), cv2.FONT_HERSHEY_COMPLEX,2.5,(200,0,200),5)

    #oyun bitmediyse topu hareket ettirme 
    else:

        #topu hareket ettir
        if ballPos[1] >= 650 or ballPos[1] <=10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        #topu çiz
        img = cvzone.overlayPNG(img,imgBall, ballPos)

        cv2.putText(img, str(score[0]),(300,650), cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),5)
        cv2.putText(img, str(score[1]),(900,650), cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),5)
        
    h, w, _ = img.shape
    img[h-130:h-10, 20:233] = cv2.resize(imgRaw,(213,120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100,100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0,0]
        imgGameOver = cv2.imread("Resources/gameOver.png")




