import numpy as np
import cv2
import pickle


# yolov3 func
def findObj(outputs, img, probLevel, NMS, classesNames):
    height, width, channel = img.shape
    boxes = []
    classIds = []
    prob = []

    for outPut in outputs:
        for det in outPut:
            tab = det[5:]
            id = np.argmax(tab)
            prediction = tab[id]
            if prediction > probLevel:
                wt = int(det[2] * width)
                ht = int(det[3] * height)
                x = int((det[0] * width - wt / 2))
                y = int((det[1] * height - ht / 2))
                boxes.append(([x, y, wt, ht]))
                classIds.append(id)
                prob.append(float(prediction))

    indices = cv2.dnn.NMSBoxes(boxes, prob, probLevel, NMS)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        wt = box[2]
        ht = box[3]
        cv2.rectangle(img, (x, y), (x + wt, y + ht), (0, 255, 0), 2)
        cv2.putText(img, f'{classesNames[classIds[i]]} {int(prob[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 255), 1)
        # boxes label option


def nothing(x):
    pass


def undistort(img, cal_dir='Resources/cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    matrix = file['mtx']
    distCo = file['dist']
    dst = cv2.undistort(img, matrix, distCo, None, matrix)
    return dst


def colorFilter(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([18,94,140])
    upperYellow = np.array([48,255,255])
    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([255, 255, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    combinedImage = cv2.bitwise_or(maskedWhite,maskedYellow)
    return combinedImage


def thresholding(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5))
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 100)
    imgDil = cv2.dilate(imgCanny,kernel,iterations=1)
    imgErode = cv2.erode(imgDil,kernel,iterations=1)
    imgColor = colorFilter(img)
    combinedImage = cv2.bitwise_or(imgColor, imgErode)

    return combinedImage,imgCanny,imgColor


def initializeTrackbars(intialTracbarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],50, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], 100, nothing)


def valTrackbars():
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    src = np.float32([(widthTop/100,heightTop/100), (1-(widthTop/100), heightTop/100),
                      (widthBottom/100, heightBottom/100), (1-(widthBottom/100), heightBottom/100)])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]) main set
    return src


def drawPoints(img,src):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]) main set
    src = src * img_size
    for x in range(0, 4):
        cv2.circle(img,(int(src[x][0]),int(src[x][1])),15,(192,66,41),cv2.FILLED)
    return img


def perspective_warp(img,
                     dest_size=(1280, 720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src * img_size
    dest = dst * np.float32(dest_size)
    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(img, M, dest_size)

    return warped

def inv_perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_win=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = get_hist(img)
    midpoint = int(histogram.shape[0] / 2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(img.shape[0] / nwindows)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # boxes loop
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if draw_win == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (255, 255, 255), 1)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (255, 255, 255), 1)
            # 0! pixels in x and y inside the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Adding to list
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # value > minpix pixels, mean next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if leftx.size and rightx.size:
        # poly
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

        # plot
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [133, 233, 96]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [133, 233, 96]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    else:
        return img,(0,0),(0,0),0


def draw_lanes(img, left_fit, right_fit,frameWidth,frameHeight,src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (110, 200, 50))
    inv_perspective = inv_perspective_warp(color_img,(frameWidth,frameHeight),dst=src)
    inv_perspective = cv2.addWeighted(img, 0.9, inv_perspective, 0.3, 0)
    return inv_perspective


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def get_nums(count):
    len_count = len(str(count))
    return ((7 - len_count) * '0') + str(count)
