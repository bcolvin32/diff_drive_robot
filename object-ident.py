import argparse
import cv2


#thres = 0.45 # Threshold to detect object

#This is to pull the information about what each object is called
classNames = []
classFile = "/home/hcolvin/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

#This is to pull the information about what each object should look like
configPath = "/home/hcolvin/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/hcolvin/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

#This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#kernel sizes for smoothing
kernelSizes = [(3, 3), (9, 9), (15, 15)]

#This is to set up what the drawn box size/color of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #Below is commented out, prints each object sighting to console
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    #box is the pre determined start (x, y) and end (x, y) values
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    #draw and label the center of the box (will be used later for 
                    #the value 60 is the y-axis value, keeps it lower than the className
                    cv2.putText(img, "boxCenter", (box[0]+10, box[1]+60), 
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    #This value is WRONG, it displayed the webcame window width
                    cv2.putText(img, str((box[2]+box[0])/2), (box[0]+200, box[1]+60), 
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    
                    
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    

    return img,objectInfo

#Below determines the size of the live feed window that will be displayed on the Raspberry pi OS
if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)

#Below is the never ending loop that determines what will happen when an object is identified
    while True:
        success, img = cap.read()
        #below provides a huge amount of control. The value 0.45 is the threshold number, the value 0.2 is the nms number
        #currently set to only detect a person
        result, objectInfo = getObjects(img,0.45,0.2, objects=['person'])
        #print(objectInfo)

        cv2.imshow("Output",img)
        cv2.waitKey(1)
        
        
