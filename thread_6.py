import sys
import cv2
import numpy as np
#th import
import threading
from tkinter import *
import tkinter as tk
root = Tk()
root.title('Hello python')
root.geometry("500x780+1450+100")
root.config(bg="SlateBlue2")#tan2

root.t0=Entry(font=("arial",25,"bold"),bg="white", bd =5, justify='center')
root.t0.place(x=52,y=250)

# FOR CHANGING THE TEXT OF THE PAUSE AND PLAY BUTTON
def fn_val_pause_play():
    global pause_play
    global stop_count
    if stop_count == 1000:
        return
    
    if pause_play ==0:
        pause_play=1
        root.b9["text"]="pause"
        root.b9["bg"]="SeaGreen1"
    elif pause_play ==1:
        pause_play =0
        root.b9["text"]="resume"
        root.b9["bg"]="tan2"

root.b9=Button(root,width=10,text='pause',command= fn_val_pause_play, font=("arial",20,"bold"),bg="SeaGreen1")
root.b9.place(x=155,y=400)


# print(type(root))
# print(type(root.t0))

#global integer for choice
ck=-1
count_object=0
prev_count_object=count_object
stop_count=0
pause_play=1
cap = cv2.VideoCapture('8_1.mp4') #bikeInsurance

whT=320
confThreshold = 0.5
nmsThreshold = 0.3

confThreshold_helmet = 0.4
nmsThreshold_helmet = 0.27

classesFile ='coco.names'
classNames = []

classesFile_helmet ='helmet.names'
classNames_helmet = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


with open(classesFile_helmet,'rt') as f:
    classNames_helmet = f.read().rstrip('\n').split('\n')


modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"

modelConfiguration_helmet = "yolov3-helmet.cfg"
modelWeights_helmet = "yolov3-helmet.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net_helmet = cv2.dnn.readNetFromDarknet(modelConfiguration_helmet, modelWeights_helmet)
net_helmet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_helmet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 
def findObjects(outputs,img,ind):
    global ck
    global count_object
    global stop_count
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    if ind == 4:
        ind = 5
    elif ind == 5:
        ind =7
    
    cn=0
    for output in outputs:
        
        
        for det in output:
            scores = det[5:]     
            classId = np.argmax(scores)
            
            
            confidence = scores[classId]
             
            if confidence > confThreshold and classId == ind :
                
                cn = cn+1
                
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                
                
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
     
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    print(str(len(indices))+ " " +classNames[ind])
    count_object=len(indices)
    print()
      
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
                   
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
            
            
        if classIds[i]==5:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0,255,0), 2)
            
            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if classIds[i]==7:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0,0,255), 2)
            
            
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
def findObjects_helmet(outputs,img):
    global ck
    global count_object
    global stop_count
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    cn = 0
    
    for output in outputs:
        
        for det in output:
            
            scores = det[5:]  
               
            classId = np.argmax(scores)
            
            confidence = scores[classId]
          
            if confidence > confThreshold_helmet and classId < 1:
              
                cn=cn+1
                
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
            
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold_helmet, nmsThreshold_helmet)

    print(str(len(indices))+ " Helmet" )
    count_object=len(indices)
    print()
    
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        if classIds[i]==0:
            cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 255), 2)
            
            cv2.putText(img,f'{classNames_helmet[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
def fn_while_1():
    global ck
    global stop_count
    global pause_play
    while True:
        
        # if stop_count == 1000:
        #     # sys.exit()
        #     print("--------------END--------------")
        #     break

        frameId = int(round(cap.get(1)))
        success, img = cap.read()
        # print(img.shape)
        if frameId % 2 == 0:
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            if ck < 6 and ck>=0:
                net.setInput(blob)
                layerNames = net.getLayerNames()
                
                outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                findObjects(outputs,img,ck)
                cv2.imshow('Image',img)      
            elif ck == 6:
                net_helmet.setInput(blob)
                layerNames = net_helmet.getLayerNames()            
                outputNames = [(layerNames[i[0] - 1]) for i in net_helmet.getUnconnectedOutLayers()]
                
                outputs = net_helmet.forward(outputNames)
                
                findObjects_helmet(outputs,img)
                
                cv2.imshow('Image',img)
            else:
                cv2.imshow('Image',img)

            while pause_play==0:
                if pause_play==1:
                    break
            
            if stop_count == 1000:
                # sys.exit()
                print("--------------END--------------")
               
                # cv2.destroyAllWindows() 
                
                # end_img=cv2.read('end.png')
                # cv2.imshow('Image', 'end.png')
                img1 = cv2.imread('/home/shyam/Desktop/img2/helmet/end_1.jpg')
                # fn, img1 = cap.read()
                cv2.imshow('Image', img1)
                cv2.waitKey(1000)
                cv2.destroyAllWindows() 
                break
                
            
        cv2.waitKey(1)
 















def fn_object_count(root):
    global count_object  
    global stop_count
    # while True:
    root.t1=Entry(font=("arial",25,"bold"),bg="white", bd=5, justify='center')
    root.t1.place(x=52,y=50)
    
    if stop_count == 1000:
        return

    # root.t0=Entry(font=("arial",18,"bold"),bg="white")
    # root.t0.place(x=100,y=350)
    # print(type(root))
    # print(type(root.t0))

    root.b1=Button(root,width=10,text='person',command=fn_val_person, font=("arial",20,"bold"),bg="cadet blue")
    root.b1.place(x=45,y=470)
    root.b2=Button(root,width=10,text='bicycle',command= fn_val_bicycle, font=("arial",20,"bold"),bg="cadet blue")
    root.b2.place(x=45,y=530)
    root.b3=Button(root,width=10,text='car',command= fn_val_car, font=("arial",20,"bold"),bg="cadet blue")
    root.b3.place(x=45,y=590)
    root.b4=Button(root,width=10,text='bike',command= fn_val_bike, font=("arial",20,"bold"),bg="cadet blue")
    root.b4.place(x=275,y=470)
    root.b5=Button(root,width=10,text='bus',command= fn_val_bus, font=("arial",20,"bold"),bg="cadet blue")
    root.b5.place(x=275,y=530)
    root.b6=Button(root,width=10,text='truck',command= fn_val_truck, font=("arial",20,"bold"),bg="cadet blue")
    root.b6.place(x=275,y=590)
    root.b7=Button(root,width=10,text='helmet',command= fn_val_helmet, font=("arial",20,"bold"),bg="cadet blue")
    root.b7.place(x=155,y=650)
    root.b8=Button(root,width=10,text='exit',command= fn_val_stop, font=("arial",20,"bold"),bg="gray40")
    root.b8.place(x=155,y=320)
    
    # root.b9["text"]="shyam"
    



def fn_put_count(): # fn to put the count of the object
    global count_object
    global prev_count_object
    if prev_count_object != count_object:
        prev_count_object=count_object
        root.t0.delete(0,'end')
        root.t0.insert(END, str(count_object))

def fn_place_count():
    global stop_count
    while True:
        if stop_count == 1000:
            root.t1.delete(0,'end')
            root.t1.insert(END, "Programm End")
            # sys.exit()            
            return
        fn_put_count()

def fn_val_stop():
    global stop_count
    stop_count = 1000


def fn_val_person():
    global ck
    ck=0
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "Person " ) #count_object
def fn_val_bicycle( ):
    global ck
    ck=1
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "bicycle " )
def fn_val_car( ):
    global ck
    ck=2
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "car " )
def fn_val_bike( ):
    global ck
    ck=3
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "bike " )
def fn_val_bus( ):
    global ck
    ck=4
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "bus " )
def fn_val_truck( ):
    global ck
    ck=5
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "truck " )
def fn_val_helmet( ):
    global ck
    ck=6
    root.t1.delete(0,'end')
    root.t1.insert(END, "Number of "+ "helmet " )
# def fn_t2_main():
#     print("t2 to main is printing")

    
def fn_while_2(): #main thread (without thread)
    global root
    print("main is called")
    fn_object_count(root)
    root.mainloop() #Run your root
























def print_triangle(num): 
    # while True:
    print("tri")
    fn_place_count()

if __name__ == "__main__": 
	# creating thread 
    t1 = threading.Thread(target=fn_while_1) 
    # t2 = threading.Thread(target=fn_while_2)
    t2 = threading.Thread(target=print_triangle, args=(10,))
    # starting thread 1 
    t1.start() 
    # starting thread 2 
    t2.start() 
    # t1.join()
    # t2.join()
    fn_while_2() # calling the main thread
    
    print("Done!") 
    
