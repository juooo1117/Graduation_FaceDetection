import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import tensorflow.keras
from tensorflow.keras import backend as K

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('/Users/parkjuhyeon/PycharmProjects/graduationProject/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('/Users/parkjuhyeon/PycharmProjects/graduationProject/dlib_face_recognition_resnet_model_v1.dat')


### 얼굴 탐지
def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=object)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)
        shapes.append(shape)

    return rects, shapes, shapes_np


### 아는 얼굴의 랜드마크 추출
def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))
    return np.array(face_descriptors)



### 인식하고 싶은 사람들 이미지 지정(사용자가 원하는 대로 바꾸면 됨)
img_paths = {'wonyoung':'wonyoung.jpg', 'rei':'rei.jpg'}

# 인식하고 싶은 사람들의 얼굴 랜드마크 추출후 'descs.npy' 라는 배열에 저장
descs = []
for name, img_path in img_paths.items():
    img = cv2.imread(img_path)
    _, img_shapes, _ = find_faces(img)
    descs.append([name, encode_faces(img, img_shapes)[0]])

np.save('descs.npy', descs)
print(descs)


### 폰트
font = cv2.FONT_HERSHEY_SIMPLEX

### 재생할 파일 (700*450)
VIDEO_FILE_PATH = 'test2.mp4'
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

# 열리는지 확인
if not cap.isOpened():
    print("cannot open the video (%d)" % VIDEO_FILE_PATH)
    exit()

titles = ['original']

# 윈도우생성 및 사이즈변경
for t in titles:
    cv2.namedWindow(t)

# 파일의 넓이/높이/frame rate 가져오기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('width {0}, height {1}, fps {2}'.format(width, height, fps))

# 저장할 비디오 코덱/이름 설정 (맥에서는 Quick time player로 avi. 파일 재생안되므로 mov.파일로 변환필요)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
filename = 'final_ive.avi'

# 최종적으로 저장할 파일(변수명: out) stream 생성
out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))


while True:
    # 파일로 부터 이미지 얻기
    ret, img = cap.read()
    if img is None:
        break;

    img = cv2.flip(img, 1)  # 좌우 변환 시켜주는 부분
    rects, shapes, _ = find_faces(img) # 얼굴 찾기
    descriptors = encode_faces(img, shapes) # 인코딩

    for i, desc in enumerate(descriptors):
        x = rects[i][0][0]  # 얼굴 X 좌표
        y = rects[i][0][1]  # 얼굴 Y 좌표
        w = rects[i][1][1] - rects[i][0][1]  # 얼굴 너비
        h = rects[i][1][0] - rects[i][0][0]  # 얼굴 높이

        # 추출된 랜드마크와 데이터베이스의 랜드마크들 중 제일 짧은 거리를 찾는 부분
        descs1 = sorted(descs, key=lambda x: np.linalg.norm([desc] - x[1]))
        dist = np.linalg.norm([desc] - descs1[0][1], axis=1)

        if dist < 0.45:  # 그 거리가 0.45보다 작다면 그 사람으로 판단
            #name = descs1[0][0]
            name = " "
        else:  # 0.45보다 크다면 모르는 사람으로 판단 -> 모자이크 처리
            name = " "
            mosaic_img = cv2.resize(img[y:y + h, x:x + w], dsize=(0, 0), fx=0.04, fy=0.04)  # 축소
            mosaic_img = cv2.resize(mosaic_img, (w, h), interpolation=cv2.INTER_AREA)  # 확대
            img[y:y + h, x:x + w] = mosaic_img  # 인식된 얼굴 영역 모자이크 처리

        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 얼굴 영역에 초록박스 쳐주는 부분
        #cv2.putText(img, str(dist)[1:6], (x + 5, y + h - 5), font, 2, (0, 0, 255), 4)  # 사진에 유사도 거리 출력해주는 부분

    # 한글
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((x + 5, y - 30), name,
            font=ImageFont.truetype("/Users/parkjuhyeon/PycharmProjects/graduationProject/batang.ttc", 15),
            fill=(255, 255, 255))  # 폰트사이즈 -> 15로 설정
    img = np.array(img)

    # 얼굴 인식된 이미지 화면 표시
    cv2.imshow(titles[0], img)

    # 인식된 이미지 파일로 저장
    out.write(img)

    # 1ms 동안 키입력 대기
    if cv2.waitKey(1) == 27:
        break;

# 재생 파일 종료
cap.release()
# 저장 파일 종료
out.release()
# 윈도우 종료
cv2.destroyAllWindows()
