import cv2
import numpy as np
import os
from scipy import spatial
import math
import mysql.connector


def Video2Image(filepath):
    times = 0
    frameFrequency = 1
    outPutDirName = 'F:\SEandprogram\TeacherLI\seproject\image'
    result = []

    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
    camera = cv2.VideoCapture(filepath)

    while True:
        times += 1
        res, image = camera.read()
        # 视频播放的实现（实现后效果并不好，视频分辨率大于屏幕）
        '''
        ret, frame = camera.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
        '''
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            cv2.imwrite(outPutDirName + '\/' + str(times) + '.jpg', image)
            print(outPutDirName + str(times) + '.jpg')
            if mark_check(times):
                print("存在路标")
                temp = picture_Process()
                result.append(temp)
            else:
                print('不存在路标')
                result.append([0, 0.0, 0.0, 0.0])

    print('over')
    camera.release()
    return result
# 图像识别
def mark_check(picture_path):
    # 定义全局cirInfo
    global cirInfo
    # 打开图片
    picturePath = ('F:\SEandprogram\TeacherLI\seproject\image\/' + str(picture_path) + '.jpg')
    img = cv2.imread(picturePath)

    # 由于图片大于我的电脑屏幕，所以只好先对图片进行缩小处理
    height, width = img.shape[:2]
    reSize = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

    # 对图片的灰度处理
    gray = cv2.cvtColor(reSize, cv2.COLOR_BGR2GRAY)
    try:
        # 霍夫曼圆判定
        circlesInImage = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 5, param1=165, param2=7, minRadius=1,
                                          maxRadius=1)
        circles = circlesInImage[0, :, :]
        circles = np.uint16(np.around(circles))
        cirInfo = circles
        if len(cirInfo) < 3 or len(cirInfo) > 7:
            return False
        # print(cirInfo)
        return True
        # 观察圆心坐标
        #for i in circles[:]:
        #    print("圆心坐标", i[0], i[1])
        # 对原图像的处理，得到可视化的处理结果
        #cv2.circle(reSize,(i[0],i[1]),i[2],(255,0,0),5)
        #cv2.circle(reSize,(i[0],i[1]),2,(255,0,255),10)
        #cv2.rectangle(reSize, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)
        #cv2.imshow("circles", reSize)

    # 对霍夫曼圆判定结果的对比
    except:
        return False
    # src = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
    # cv2.namedWindow("input image",cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input image",src)


# 对图处理后的cirInfo进行处理
def picture_Process():
    # 设置函数内变量，处理cirInfo

    # 没得未来的穷举
    """
    dis_m = 0
    for p1 in range(len(cirInfo)):
        point1 = cirInfo[p1]
        for p2 in range(p1 + 1, len(cirInfo)):
            point2 = cirInfo[p2]
            #print(point1[0])
            #print(point2[0])
            dis = distance(point1,point2)
            dis_m = max(dis_m, dis)
    print(dis_m)
    """
    # 先进的凸包算法(求距离最远的两个点

    candidates = cirInfo
    dist_mat = spatial.distance_matrix(candidates, candidates)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    # 计算航向角，中心的相对坐标
    dX = (float(candidates[i][0]) + float(candidates[j][0])) / 2
    dY = (float(candidates[i][1]) + float(candidates[j][1])) / 2

    # 这个地方，凸包给的结果太复杂，不好调用，索性直接写(还有优化空间
    def distance(point1, point2):
        return pow(pow(float(point1[0]) - float(point2[0]), 2) + pow(float(point1[1]) - float(point2[1]), 2), 0.5)

    dis_max = distance(candidates[i], candidates[j])
    disMid = 0.5 * dis_max
    point = i
    # 寻找路标的顶点（坐标系的原点）
    for p in range(len(candidates)):
        if int(distance(candidates[p], [dX, dY])) in range(int(disMid) - 7, int(disMid) + 7):
            if p != i and p != j:
                point = p

    # print(i, j, point)
    dis_angle = distance(candidates[point], [dX, dY])
    aX = float(dX) - float(candidates[point][0])
    aY = float(dY) - float(candidates[point][1])

    # print(candidates[point])
    # print(dX, dY)
    angcos = aX / dis_angle
    angsin = aY / dis_angle

    # 计算航向角
    nowangle = 0
    if angcos > 0 and angsin < 0:
        angle = math.acos(angcos) * 180 / math.pi + 45.0
        nowangle = angle
    elif angcos < 0 and angsin < 0:
        angle = math.acos(angcos) * 180 / math.pi + 45.0
        nowangle = angle
    elif angcos < 0 and angsin > 0:
        angle = math.acos(angcos) * 180 / math.pi + 225.0
        nowangle = angle
    elif angcos > 0 and angsin > 0:
        if math.acos(angcos) * 180 / math.pi < 45.0:
            angle = 45 - math.acos(angcos) * 180 / math.pi
            nowangle = angle
        elif math.acos(angcos) * 180 / math.pi > 45.0:
            angle = 405 - math.acos(angcos) * 180 / math.pi
            nowangle = angle
    elif angcos == 0 and angsin == 1:
        angle = 135.0
        nowangle = angle
    elif angcos == 0 and angsin == -1:
        angle = 315.0
        nowangle = angle
    elif angcos == 1:
        angle = 45.0
        nowangle = angle
    elif angcos == -1:
        angle = 225.0
        nowangle = angle
    # print (nowangle)
    # 识别路标的ID，test01的ID命名于设计不同，暂定为101
    # 根据三个定位点得到直角坐标系，计算相对原点坐标
    '''
    PicInfo = []
    for m in range(len(candidates)):
        PicInfo.append([float(candidates[m][0]) - int(candidates[point][0]),float(candidates[m][1]) - int(candidates[point][1])])
    '''
    # ID计算（有bug,原因是图像识别的时候往往不能识别的很准确，导致有误差，在使用距离判断时就会有疏漏,可以改进
    count1 = 0
    count2 = 0
    count3 = 0
    for m in range(len(candidates)):
        if m != i and m != j and m != point:
            if int(distance(candidates[m], candidates[point])) in range(12, 19):
                count1 += 1
            elif int(distance(candidates[m], candidates[point])) in range(34, 39):
                count3 += 1
            elif int(distance(candidates[m], candidates[point])) in range(28, 36):
                count2 += 1
            elif int(distance(candidates[m], candidates[point])) in range(18, 28):
                count2 += 1

    mark_ID = 0
    if count1 == 2:
        mark_ID = mark_ID + 100
    if count2 == 3:
        mark_ID = mark_ID + 10
    if count3 == 2:
        mark_ID = mark_ID + 1

    # print(count1,count2,count3)
    # print('dX=', dX)
    # print('dY=', dY)
    # print('航向角=', nowangle)
    # print('ID =', mark_ID)
    mark_Id = 101
    # save_process(dX, dY, nowangle, mark_Id)
    result = [mark_Id, dX, dY, nowangle]
    return result



# 构建数据库proce_result
def make_DB_reuslt():
    resultDB = mysql.connector.connect(
        host="localhost",
        user='root',
        passwd='17932486'
    )
    resultsor = resultDB.cursor()
    try:
        resultsor.execute("CREATE DATABASE result_db")
    except:
        print('created')
    try:
        resultDB = mysql.connector.connect(
            host="localhost",
            user='root',
            passwd='17932486',
            database='result_db'
        )
        resultsor = resultDB.cursor()
        resultsor.execute("DROP TABLE IF EXISTS result")
        resultsor.execute("CREATE TABLE result (ID INT AUTO_INCREMENT PRIMARY KEY, "
                          "mark_ID INT, "
                          "dX INT, "
                          "dY INT, "
                          "nowangle FLOAT)")
    except:
        print("created")
    resultsor.close()
    resultDB.close()


def make_DB_loadmark():
    markDB = mysql.connector.connect(
        host="localhost",
        user='root',
        passwd='17932486'
    )
    marksor = markDB.cursor()
    try:
        marksor.execute("CREATE DATABASE loadmark_db")
    except:
        print("created")
    try:
        markDB = mysql.connector.connect(
            host='localhost',
            user='root',
            passwd='17932486',
            database='loadmark_db'
        )
        marksor = markDB.cursor()
        marksor.execute("DROP TABLE IF EXISTS mark")
        marksor.execute("CREATE TABLE mark("
                        "ID INT AUTO_INCREMENT PRIMARY KEY,"
                        "MARK_ID INT,"
                        "PreX INT,"
                        "PreY INT)")
    except:
        print('created')
    marksor.close()
    markDB.close()


def save_process(markId, dX, dY, nowangle):
    resultDB = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='17932486',
        database='result_db'
    )
    resultsor = resultDB.cursor()

    sql = "INSERT INTO result (mark_ID, dX, dY, nowangle) VALUES(%s, %s, %s, %s)"
    val = [markId, dX, dY, nowangle]
    resultsor.execute(sql, val)

    resultDB.commit()

    resultsor.close()
    resultDB.close()


def pre_Mark():
    markDB = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='17932486',
        database='loadmark_db'
    )

    marksor = markDB.cursor()
    sql = "INSERT INTO mark (mark_ID, PreX, PreY) VALUES(%s, %s, %s)"
    val = [101, 480, 270]
    marksor.execute(sql, val)
    markDB.commit()

    marksor.close()
    markDB.close()
