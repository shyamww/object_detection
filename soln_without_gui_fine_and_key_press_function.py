import cv2
import numpy as np
# import keyboard

cap = cv2.VideoCapture('v7.mp4')
whT=320
confThreshold = 0.5
nmsThreshold = 0.3


ck=2


#helmet
confThreshold_helmet = 0.4
nmsThreshold_helmet = 0.27

classesFile ='coco.names'
classNames = []
#helmet
classesFile_helmet ='helmet.names'
classNames_helmet = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
#helmet
with open(classesFile_helmet,'rt') as f:
    classNames_helmet = f.read().rstrip('\n').split('\n')

## Model Files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"
#helmet
modelConfiguration_helmet = "yolov3-helmet.cfg"
modelWeights_helmet = "yolov3-helmet.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#helmet
net_helmet = cv2.dnn.readNetFromDarknet(modelConfiguration_helmet, modelWeights_helmet)
net_helmet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_helmet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def fn_key():
    global ck
    x=(input("person=0 bicycle=1 car=2 bike=3 bus=4 truck=5 helmet=6:--"))
    if len(x)==1:
        ck=int(x[0])
    else:
        ck=int(x[len(x)-1])
    # print("hiiiiiiiiii")




def findObjects(outputs,img,ind):
    global ck
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    if ind == 4:
        ind = 5
    elif ind == 5:
        ind =7
    # print(ind)
    cn=0
    for output in outputs:
        # if keyboard.is_pressed('q'):
        #     fn_key()
        for det in output:
            scores = det[5:]     
            classId = np.argmax(scores)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            confidence = scores[classId]
                
            # if keyboard.is_pressed('q'):
            #     fn_key()
            # if confidence>0:
            #     print(str(confidence) + " "+ str(classId) +" "+ str(ind))
            if confidence > confThreshold and classId == ind :
                # print("hii")
                cn = cn+1
                # print(str(classId) + " " + str(ck[classId]) +" "+ str(classNames[classId]))
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                # if keyboard.is_pressed('q'):
                #     fn_key()
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                # if keyboard.is_pressed('q'):
                #     fn_key()
    # if keyboard.is_pressed('q'):
    #     fn_key()
    print(str(cn)+ " " + classNames[ind])
    # if keyboard.is_pressed('q'):
    #     fn_key()
    print()
        
    # if keyboard.is_pressed('q'):
    #     fn_key()
    
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        
    # if keyboard.is_pressed('q'):
    #     fn_key()
        
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # if keyboard.is_pressed('q'):
        #     fn_key()
            
        if classIds[i]==0:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 255, 0), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if classIds[i]==1:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0 , 255), 2)
        if classIds[i]==2:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255 , 255), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255 , 255), 2)        
        if classIds[i]==3:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            
        if classIds[i]==5:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0,255,0), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if classIds[i]==7:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0,0,255), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # if keyboard.is_pressed('q'):
        #     fn_key()


#helmet
def findObjects_helmet(outputs,img):
    global ck
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    cn = 0
    # for i in range(1):
    #     cn.append(0)
    for output in outputs:
        
        # if keyboard.is_pressed('q'):
        #     fn_key()
        
        for det in output:
            
            # print("ajhsdjsa")
            scores = det[5:]  
                
            # if keyboard.is_pressed('q'):
            #     fn_key()
            classId = np.argmax(scores)
            # print(classId)
            confidence = scores[classId]
            # if confidence>0:
            #     print(confidence)
            if confidence > confThreshold_helmet and classId < 1:
              
                # cn[classId]=cn[classId]+1
                cn=cn+1
                # print(str(classId) + " " + str(ck[classId]) +" "+ str(classNames_helmet[classId]))
                
                
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
            #     if keyboard.is_pressed('q'):
            #         fn_key()
            # if keyboard.is_pressed('q'):
            #     fn_key()

    # for x in range(len(cn)):
    # if keyboard.is_pressed('q'):
    #     fn_key()
    print(str(cn)+ " Helmet")
    # if keyboard.is_pressed('q'):
    #     fn_key()
    print()
    
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold_helmet, nmsThreshold_helmet)
    # if keyboard.is_pressed('q'):
    #     fn_key()
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # if keyboard.is_pressed('q'):
        #     fn_key()
        # print(x,y,w,h)
        if classIds[i]==0:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 255), 2)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            cv2.putText(img,f'{classNames_helmet[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # if keyboard.is_pressed('q'):
        #     fn_key()

# ck=-1
while True:
    # if keyboard.is_pressed('q'):
    #     fn_key()
    frameId = int(round(cap.get(1)))
    # if keyboard.is_pressed('q'):
    #     fn_key()
    success, img = cap.read()
    print(img.shape)
    # if keyboard.is_pressed('q'):
    #     fn_key()
        # x=(input("person=0 bicycle=1 car=2 bike=3 bus=4 truck=5 helmet=6"))
        # if len(x)==1:
        #     ck=int(x[0])
        # else:
        #     ck=int(x[1])
    #network understand images only in the form of blob
    # print(ck)
    if frameId % 3 == 0:
        # if keyboard.is_pressed('q'):
        #     fn_key()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        # if keyboard.is_pressed('q'):
        #     fn_key()
        #if keyboard.is_pressed('q'):        
        if ck < 6 and ck>=0:
            net.setInput(blob)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            layerNames = net.getLayerNames()
            # print(layerNames)
            outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
            # print(outputNames)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            outputs = net.forward(outputNames)
            # print(outputs)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            findObjects(outputs,img,ck)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            cv2.imshow('Image',img)      
            # if keyboard.is_pressed('q'):
            #     fn_key()
        # cv2.waitKey(2000)           
        elif ck == 6:
            net_helmet.setInput(blob)
            # if keyboard.is_pressed('q'):
            #         fn_key()
            layerNames = net_helmet.getLayerNames()
            # print(layerNames)
            outputNames = [(layerNames[i[0] - 1]) for i in net_helmet.getUnconnectedOutLayers()]
            # print(outputNames)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            
            outputs = net_helmet.forward(outputNames)
            # print(outputs)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            
            findObjects_helmet(outputs,img)
            # if keyboard.is_pressed('q'):
            #     fn_key()
            cv2.imshow('Image',img)      
            # if keyboard.is_pressed('q'):
            #     fn_key()
        else:
            cv2.imshow('Image',img)  
        #     if keyboard.is_pressed('q'):
        #         fn_key()   
        #     if keyboard.is_pressed('q'):
        #         fn_key()             
        # if keyboard.is_pressed('q'):
        #         fn_key()
    cv2.waitKey(1)
    


