import cv2
cap = cv2.videoCapture()
while True :
  ret, frame cap.read() # read frame/image one by one
  resized = cv2.resize(frame, (600,400))
  cv2.imshow("Frame", resized) # display frame/ image
  key = cv2.waitKey(1) # wait till key press
  if key == ord("q"): # exit Loop on 'q' key press
    break
cap. release() # release video capture object
cv2 . destroyA11Windows() # destroy aLL frame windows
