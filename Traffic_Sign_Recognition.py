import numpy as np
import pandas as pd
import cv2 as cv
import random
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential # type: ignore
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout # type: ignore
from keras.optimizers import Adam, RMSprop, Adagrad # type: ignore

## 이미지 불러오기, 라벨링 작업 ##
def Image_Label(Image_root, Label_root, image_list, label_list): # 이미지 불러오기와 라벨링 작업
    csv = pd.read_csv(Label_root)
    for image_name in os.listdir(Image_root):
        if image_name == '.DS_Store': # 이미지 이름 중에 이상한 이름이 저장되어 있음
            continue
        
        image = cv.imread(os.path.join(Image_root, image_name))
        image_list.append(image)
        label_list.append(int(csv[csv['Image'] == image_name]['Label'].iloc[0])) # OpenCV의 os.listdir 함수는 무작위로 불러오기 때문에 csv 라벨을 이미지와 동일한 라벨을 불러와야하기 때문

        for i in range(max(label_list) + 1): # 이상치로 간주된 이미지들을 제거한 후의 이미지들 중 라벨 당 이미지 개수를 동일화 하기 위해 가장 적은 146개를 기준으로 초과할 시 제거
            if label_list.count(i) > 1483: 
                image_list.pop()
                label_list.pop()

## 데이터 전처리 ##
def Brightness(images, standard): # 밝기 조절
    for index, image in enumerate(images):
        hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        # V 채널 추출
        h, s, v = cv.split(hsv)
        v_mean = np.mean(v)

        # standard일 경우 일정 값만 추가, brighten_ratio일 경우 기준 밝기에 맞춰서 조절
        brighten_ratio = standard - v_mean # 하나의 이미지의 V 채널의 값을 얼마나 더해야하는지
        v = cv.add(v, brighten_ratio) # 하나의 밝기를 기준으로 할 경우 위의 주석과 함께 사용
        # v = cv.add(v, standard) # V 채널에 차이값을 더해줌

        # V 채널이 0~255 범위를 넘지 않도록 조절
        v = np.clip(v, 0, 255)

        # HSV 채널 결합
        hsv = cv.merge([h, s, v])
        images[index] = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

def Resize_image(images): # 리사이징
    for index, image in enumerate(images):
        images[index] = cv.resize(image, dsize = d_size, interpolation = cv.INTER_LINEAR)

## 이미지 데이터 증강 ##
# 노이즈 증강
def Noise_Augmentation(image_list, label_list, label, section):
    list = [image_list[index] for index, value in enumerate(label_list) if value == label]

    for i in range(section):
        image = list[i % 100]

        kernel_size = random.choice([(1, 1), (3, 3), (5, 5)])
        # kernel_size = (1, 1)
        image = cv.blur(image, kernel_size)

        image_list.append(image)
        label_list.append(label)

# 가우시안 블러 증강
def Gaussian_Augmentation(image_list, label_list, label, section):
    list = [image_list[index] for index, value in enumerate(label_list) if value == label]

    for i in range(section):
        image = list[i % 100]

        sigma = random.uniform(0, 1)
        # sigma = 0.5
        image = cv.GaussianBlur(image, (0, 0), sigma)

        image_list.append(image)
        label_list.append(label)

## 신경망 모델 설계 및 실행, 평가 ##
def Model_of_CNN(): # 모델 설계 함수
    model = Sequential()
    model.add(Input(shape = train_image[0].shape))
    model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Dropout(0.6))
    model.add(Flatten())

    model.add(Dense(86, activation = 'relu'))
    model.add(Dropout(0.3)) # 학습과 테스트 이후 Dropout값 조정하기, 과적합을 방지하기 위해 Dense 층 사이에 추가
    model.add(Dense(43, activation = 'softmax'))
    return model

def Compile_and_Learning(model, validation_data, batch_siz, epoch): # 컴파일 및 학습 함수
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    hist = model.fit(train_image, train_label, batch_size = batch_siz, epochs = epoch, validation_data = validation_data, verbose = 2)
    res = model.evaluate(test_image, test_label, verbose = 0)
    print('정확도 : ', res[1]*100)
    return hist

## 혼동 행렬 그래프 ##
def calculate_confusion_matrix(true_labels, predicted_labels, num_classes):
    # 혼동 행렬 초기화
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 혼동 행렬 계산
    for t, p in zip(true_labels, predicted_labels):
        conf_matrix[t, p] += 1
    
    return conf_matrix

def plot_confusion_matrix(conf_matrix, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 혼동 행렬을 이미지로 표시
    cax = ax.matshow(conf_matrix, cmap='Blues')
    
    # 색상 바 추가
    fig.colorbar(cax)
    
    # 라벨 설정
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    
    plt.title('Confusion Matrix')
    plt.show()

def plot_confusion_matrix_with_labels(conf_matrix, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 혼동 행렬을 이미지로 표시
    cax = ax.matshow(conf_matrix, cmap='Blues')
    
    # 색상 바 추가
    fig.colorbar(cax)
    
    # 라벨 설정
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    
    # 각 셀에 예측된 값 표기 (참값과 예측값이 다른 경우에도 숫자 표시)
    for i in range(num_classes):
        for j in range(num_classes):
            if conf_matrix[i, j] > 0:  # 혼동 행렬 값이 0이 아닌 경우에만 숫자 표기
                ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='red')

    plt.title('Confusion Matrix with Labels')
    plt.show()


## main ##
start_time = time.time() # 실행 시작 시간
Train_image_root = "/Users/iseungju/Desktop/Image/16*16</train_image"
Train_label_root = "/Users/iseungju/Desktop/Image/16*16</train_label.csv"
Test_image_root = "/Users/iseungju/Desktop/Image/16*16</test_image"
Test_label_root = "/Users/iseungju/Desktop/Image/16*16</test_label.csv"

train_image = [] # '32 * 32 <'일 경우 13352 개 / '16 * 16 <'일 경우 26300개 / 라벨 당 이미지 분류 후 6278개 / 증강 기준인 1,000개를 기준으로 할 경우 23,279개
train_label = [] # '32 * 32 <'일 경우 13352 개 / '16 * 16 <'일 경우 26300개 / 라벨 당 이미지 분류 후 6278개 / 증강 기준인 1,000개를 기준으로 할 경우 23,279개
test_image = [] # '32 * 32 <'일 경우 6160 개 / '16 * 16 <'일 경우 12446 개
test_label = [] # '32 * 32 <'일 경우 6160 개 / '16 * 16 <'일 경우 12446 개

d_size = (61, 61) # dsize가 입력 받는 순서 (width, height)
standard = 130 # 밝기 처리 기준점

# 이미지와 라벨링 불러오기
Image_Label(Train_image_root, Train_label_root, train_image, train_label)
Image_Label(Test_image_root, Test_label_root, test_image, test_label)

## Train 이미지 전처리
# Bilateral_filter(train_image)
Brightness(train_image, standard)
Resize_image(train_image)
# Normalize(train_image)

## Test 이미지 전처리
# Bilateral_filter(test_image)
Brightness(test_image, standard)
Resize_image(test_image)
# Normalize(test_image)

## 이미지 데이터 증강
for label in range(max(train_label) + 1):
    section = int((1000 - train_label.count(label))/2) # 4가지의 증강 방법을 사용하는데 하나의 증강 방법에 대해서 증강할 이미지 개수들
    alpha = (1000 - train_label.count(label))%2 # 이미지 증강하는 과정에서 정확한 1,000개를 맞추기 위한 개수

    # Brightness_Augmentation(train_image, train_label, label, section + alpha)
    # Contrast_Augmentation(train_image, train_label, label, section)
    Noise_Augmentation(train_image, train_label, label, section + alpha)
    Gaussian_Augmentation(train_image, train_label, label, section)


# 신경망 모델에 입력하기 위해 train_image 데이터를 numpy 배열로 변환
train_image = np.array(train_image)
test_image = np.array(test_image)

# label에 저장되어 있는 라벨들을 원핫코드로 변환
train_label = tf.keras.utils.to_categorical(train_label, 43)
test_label  = tf.keras.utils.to_categorical(test_label , 43)

batch_siz = 64
epoch = 30
validation_data = (test_image, test_label)
# optimizers = Adam

model = Model_of_CNN() # CNN 신경망 모델 생성
model.summary() # 모델의 구조 확인
hist = Compile_and_Learning(model, validation_data, batch_siz, epoch) # CNN 신경망 모델 컴파일 및 학습

# 실행 시간 체크
end_time = time.time() # 실행 종료 시간
loading_time = end_time - start_time
print('실행 시간 : ', loading_time)

## 정확도, 손실 그래프 그리기 ##
# 정확도 그래프
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()

# 손실값 그래프
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()

# 모델을 사용해 테스트 데이터 예측
predicted_probs = model.predict(test_image)
predicted_labels = np.argmax(predicted_probs, axis=1)

# 실제 라벨은 원핫 인코딩에서 다시 라벨로 변환
true_labels = np.argmax(test_label, axis=1)

# 혼동 행렬 계산
conf_matrix = calculate_confusion_matrix(true_labels, predicted_labels, num_classes=43)

# 혼동 행렬 시각화
plot_confusion_matrix(conf_matrix, num_classes=43)


# 혼동 행렬 계산
conf_matrix = calculate_confusion_matrix(true_labels, predicted_labels, num_classes=43)

# 혼동 행렬 시각화 (각 셀에 숫자 추가)
plot_confusion_matrix_with_labels(conf_matrix, num_classes=43)