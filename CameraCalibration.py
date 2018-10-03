#카메라 캘리브레이션 수행 파일
#평행이 잘맞아도 다양한 왜곡 요인에 의해 행정렬된 영상을 잘 얻지 못하므로 캘리브레이션 수행
#캘리브레이션 후 파라미터들은 npz파일로 저장

import cv2
import numpy as np

#캘러브레이션 하는 메소드
def saveCamCalibration():
    #캘리브레이션 하기전 세팅
    #(8,5)-> chess이미지의 격자수와 동일
    termination=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30,0.001)
    objp=np.zeros((8*5,3), np.float32)
    objp[:,:2]=np.mgrid[0:8, 0:5].T.reshape(-1,2)

    #캘리브레이션 후 필요 파라미터들을 저장되는 변수
    objpoints=[]
    imgpoints=[]

    count=0
    while True:
        #이미지 불러온 후 그레이스케일로 바꿈
        img= cv2.imread("./dataset_split_v2/10_03_chessboard_s2.jpg")
        gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #chessboard에 사각형 코너를 찾음
        #rec - True:사각형 있음, False:사각형없음
        ret, corner=cv2.findChessboardCorners(gray, (8,5), None)

        print(ret)
        print(corner)

        #사각형 정보가 있으면 해당좌표 objpoints, imgpoints에 추가
        if ret:
            #objpoints추가
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corner, (11,11), (-1,-1), termination)
            imgpoints.append(corner)

            #화면에 corner정보 그림
            cv2.drawChessboardCorners(img, (8,5), corner, ret)
            count+=1
            print('[%d]'%(count))

        #이미지 사이즈가 너무크기때문에 resize수행후 이미지 show
        imS = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        cv2.imshow('img', imS)

        cv2.waitKey(0)

        if count==1:
            break
        '''
        if k==27:
            print("log 8")
            break
        '''

    cv2.destroyAllWindows()

    #파라미터 정보 불러옴
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    #파라미터 정보 저장 - npz파일로 
    np.savez('calib_s2.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    #np.save('calib.xml', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print("camera calibration data saved")

#카메라 캘리브레이션 수행
saveCamCalibration()


#npz 파일을 읽어와 해당 키의 파라미터 정보들을 출력
test = np.load('calib_s1.npz')
print(test.files)

print('ret')
print(test['ret'])

print('mtx')
print(test['mtx'])

print('dist')
print(test['dist'])

print('rvecs')
print(test['rvecs'])

print('tvecs')
print(test['tvecs'])

