import cv2
import numpy as np

cap = cv2.VideoCapture('sherbrooke_video.avi')
whT=320
confThreshold = 0.5
nmsThreshold = 0.3



classesFile ='coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

## Model Files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    ck = []
    for i in range(80):
        ck.append(0)
    for output in outputs:
        

        # ck = []
        # for i in range(80):
        #     ck.append(0)
        for det in output:
            
            # print("ajhsdjsa")
            scores = det[5:]  
                             
            classId = np.argmax(scores)
            
            confidence = scores[classId]
            if confidence > confThreshold and classId < 4:
              
                ck[classId]=ck[classId]+1
                # print(str(classId) + " " + str(ck[classId]) +" "+ str(classNames[classId]))
                
                
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))


               
                # for x in range(len(ck)):
                #     if ck[x]>0:
                #         print(str(ck[x])+" "+str(classNames[x]))
    # print(str(classId) + " " + str(ck[classId]) +" "+ str(classNames[classId]))
    
    #printing the number of vehicals and number of persons
    for x in range(len(ck)):
        if ck[x]>0:
            print(str(ck[x])+ " " + classNames[x])
    print()
    
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        if classIds[i]==0:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 255, 0), 2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if classIds[i]==1:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0 , 255), 2)
        if classIds[i]==2:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255 , 255), 2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255 , 255), 2)
        if classIds[i]==3:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)




while True:
    frameId = int(round(cap.get(1)))
    success, img = cap.read()
   
    #network understand images only in the form of blob
    if frameId % 10 == 0:
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        
        layerNames = net.getLayerNames()
        # print(layerNames)
        outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        # print(outputNames)
        
        outputs = net.forward(outputNames)
        # print(outputs)
        
        findObjects(outputs,img)
        cv2.imshow('Image',img)      
                
  
    cv2.waitKey(1)
    
    