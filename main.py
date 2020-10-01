import numpy as np
import cv2
import LaneDetection
import glob
import os

print("Program WayTracker V1")
print("Wpisz '1' jesli chcesz przejsc do przetwarzania filmu live (4fps/s)")
print("Wpisz liczbe rozna od 1 aby wygenerowac film 30fps/s")

choice = input("Podaj liczbe ")
if(int(choice) == 1):

    camFeed = False
    videoPath = 'Resources/test1.mp4'
    camNum = 1
    frameWidth = 640
    frameHeight = 480

    # yolobegin
    classesPath = 'Resources/coco.names'
    with open(classesPath, 'rt') as f:
        classesNames = f.read().rstrip('\n').split('\n')

    #print(classesNames)
    #print(len(classesNames))
    mCfg = 'Resources/yolov3.cfg'
    mWei = 'Resources/yolov3W.weights'
    yolo = cv2.dnn.readNetFromDarknet(mCfg, mWei)
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    WH = 160
    probLevel = 0.5
    NMS = 0.3


    if camFeed:intialTracbarVals = [40, 74, 26, 97] #  #wT,hT,wB,hB
    else:intialTracbarVals = [40,64,15,97]

    if camFeed:
        cap = cv2.VideoCapture(camNum)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
    else:
        cap = cv2.VideoCapture(videoPath)
    count=0
    LaneDetection.initializeTrackbars(intialTracbarVals)

    while True:
        success, img = cap.read()
        if camFeed == False: img = cv2.resize(img, (frameWidth, frameHeight), None)
        imgWarpPoints = img.copy()
        imgFinal = img.copy()
        imgCanny = img.copy()

        #yolov3 part
        convert = cv2.dnn.blobFromImage(img, 1/255, (WH, WH), [0, 0, 0], 1, crop=False)
        yolo.setInput(convert)

        layerNames = yolo.getLayerNames()
        #print(layerNames)

        outPutNames = [layerNames[i[0]-1] for i in yolo.getUnconnectedOutLayers()]
        outPut = yolo.forward(outPutNames)
        #print(type(outPut))

        LaneDetection.findObj(outPut, img, probLevel, NMS, classesNames)
        cv2.waitKey(1)

        imgUndistort = LaneDetection.undistort(img)
        imgThresh, imgCanny, imgColor = LaneDetection.thresholding(imgUndistort)
        src = LaneDetection.valTrackbars()
        imgWarp = LaneDetection.perspective_warp(imgThresh, dest_size=(frameWidth, frameHeight), src=src)
        imgWarpPoints = LaneDetection.drawPoints(imgWarpPoints, src)
        imgSliding, curves, lanes, ploty = LaneDetection.sliding_window(imgWarp, draw_win=True)

        try:
            imgFinal = LaneDetection.draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)

        except:
            lane_curve = 0
            pass

        imgStacked = LaneDetection.stackImages(0.5, ([img,imgUndistort,imgWarpPoints],
                                             [imgColor, imgCanny, imgThresh],
                                             [imgWarp,imgSliding,imgFinal]
                                             ))

        cv2.imshow("Analize screens",imgStacked)
        cv2.imshow("DetectionCam", imgFinal)
        cv2.imwrite(f'Resources/ProcessedVid/ProcImg0000{LaneDetection.get_nums(count)}.jpg', imgFinal)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    #array for video result
    processedImgArray = []

    for filename in glob.glob('Resources/ProcessedVid/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        processedImgArray.append(img)

    for filename in glob.glob('Resources/ProcessedVid/*.jpg'):
        os.remove(filename)

    out = cv2.VideoWriter('D:/Program Files/PythonProjects/WayTracker/Resources/WayTrackerResult.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)


    for i in range(len(processedImgArray)):
        out.write(processedImgArray[i])
    out.release()
    print("Przetowrzony film (WayTrackerResult.avi) znajduje sie w katalogu /Resources")
