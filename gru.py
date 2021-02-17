from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from keras.losses import mean_squared_error
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt



train = pd.read_csv(r'C:\Users\MIN\Desktop\AIFrenz_Season1_dataset\train.csv')
test = pd.read_csv(r'C:\Users\MIN\Desktop\AIFrenz_Season1_dataset\test.csv')
train1 = train.drop(['id'], axis=1)
test1 = test.drop(['id'], axis=1)

# (행의개수, 열의개수)
print(train1.shape)
print(test1.shape)
# 클래스 유형, 행 인덱스 구성, 열이름의 종류와 개수, 각 열의 자료형과 개수, 메모리햘당량정보
print(train1.info())
print(test1.info())
# 기술통계정보(평균, 표준편차, 최대값, 최소값, 중간값)
print(train1.describe())
print(test1.describe())
# 결측확인
train.isnull().sum()
# 중복확인
train.duplicated().sum()

temperature = ["X00", "X07", "X28", "X31", "X32"]   # 기온
localpress = ["X01", "X06", "X22", "X27", "X29"]    # 현지기압
speed = ["X02", "X03", "X18", "X24", "X26"]         # 풍속
water = ["X04", "X10", "X21", "X36", "X39"]         # 일일 누적강수량
press = ["X05", "X08", "X09", "X23", "X33"]         # 해면기압
sun = ["X11", "X14", "X16", "X19", "X34"]           # 일일 누적일사량
humidity = ["X12", "X20", "X30", "X37", "X38"]      # 습도
direction = ["X13", "X15", "X17", "X25", "X35"]     # 풍향

ylocal = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06',
          'Y07', 'Y08', 'Y09', 'Y10', 'Y11', 'Y12',
          'Y13', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18']
train[ylocal].head()

# plot 그리기 plot(xvalue , yvalue, figsize=(가로,세로), 제목)
train.plot(x='id', y=temperature, figsize=(8, 3), title="temperature")
train.plot(x='id', y=localpress, figsize=(8, 3), title="localpress")
train.plot(x='id', y=speed, figsize=(8, 3), title="speed")
train.plot(x='id', y=water, figsize=(8, 3), title="water")
train.plot(x='id', y=press, figsize=(8, 3), title="press")
train.plot(x='id', y=sun, figsize=(8, 3), title="sun")
train.plot(x='id', y=humidity, figsize=(8, 3), title="humidity")
train.plot(x='id', y=direction, figsize=(8, 3), title="direction")

# 예를들어 풍속과 습도 평균 표준편차값
print(train[speed])
print(train[speed].mean())
print(train[speed].std())
print('\n')
print(train[humidity])
print(train[humidity].std())


# y00 ~ y18 temperature
train.loc[:, ylocal].plot(figsize=(20, 10))

# X00 ~ X39 변수와 Y00 ~ Y17의 상관관계 및 히트맵
heatmap = train.iloc[:, 1:-1]
corr = heatmap.corr()
plt.figure(figsize=(60, 30))
sns.heatmap(corr, cmap="RdYlGn", annot=True, vmin=0, vmax=1)

# X00~ X39 변수와 Y18끼리의 상관관계 및 히트맵
heatmap2 = train.iloc[4320:, 1:40]
heatmap3 = train.iloc[:, -1:]
heatmap = pd.concat([heatmap2, heatmap3], axis=1)
corr = heatmap.corr()
plt.figure(figsize=(60, 30))
sns.heatmap(corr, cmap=plt.cm.RdYlBu_r, annot=True, vmin=0, vmax=1)

print(heatmap)

# Y00 ~ Y17의 상관관계 및 히트맵
y17 = ylocal[:-1]
corr = train[y17].corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, cmap="RdYlGn", annot=True, vmin=0, vmax=1)

# 표준화
def standard(df):
    mean = np.mean(df)
    std = np.std(df)
    norm = (df - mean) / (std - 1e-07)
    return norm, mean, std
tnorm, tmean, tstd = standard(train[1:40])
print(tnorm)
print(tmean)
print(tstd)

# 표준화한 새로운 데이터셋
train2 = pd.concat([train["id"], tnorm], axis=1)
print(train)
print(train2)

# 범주형(카데고리) 데이터 처리   온도, 습도, 풍속
# "X00","X07","X28","X31","X32"

count, bin_dividers = np.histogram(train[temperature], bins=3)
print(bin_dividers)

bin_names = ['저온', '보통', '고온']

train['hp_bin00'] = pd.cut(x=train['X00'],      # 데이터 배열
                           bins=bin_dividers,   # 경계 값 리스트
                           labels=bin_names,    # bin 이름
                           include_lowest=True) # 첫 경계값 포함

train['hp_bin07'] = pd.cut(x=train['X07'],      # 데이터 배열
                           bins=bin_dividers,   # 경계 값 리스트
                           labels=bin_names,    # bin 이름
                           include_lowest=True) # 첫 경계값 포함

print(train[['X00', 'hp_bin00', 'X07', 'hp_bin07']].sample(3))
print("\n")
temperature_dummies = pd.get_dummies(train['hp_bin00'])
print(temperature_dummies)

# 히트맵으로 파악한 고장난 센서 버리기
sensor = ['X14', 'X16', 'X19']
train.drop(sensor, axis=1, inplace=True)
test.drop(sensor, axis=1, inplace=True)

# 풍향 버리기
drop_wind = ['X13', 'X15', 'X17', 'X25', 'X35']
train.drop(drop_wind, axis=1, inplace=True)
test.drop(drop_wind, axis=1, inplace=True)

print(train.info())

print(train['X11'].head(100))
print(train['X34'])

# 누적일조량->단순일조량
def my_sun(train):
    temp = train.iloc[:, np.where(train.columns.str.find('X11') == 0)[0]]
    temp2 = temp.copy()
    for i in range(1, len(temp)):
        temp2.iloc[i, :] = temp.iloc[i, :] - temp.iloc[i - 1, :]
    temp2.iloc[np.where(temp2.sum(axis=1) < 0)[0], :] = 0
    train['X11'] = temp2['X11']

    temp = train.iloc[:, np.where(train.columns.str.find('X34') == 0)[0]]
    temp2 = temp.copy()
    for i in range(1, len(temp)):
        temp2.iloc[i, :] = temp.iloc[i, :] - temp.iloc[i - 1, :]
    temp2.iloc[np.where(temp2.sum(axis=1) < 0)[0], :] = 0
    train['X34'] = temp2['X34']

    return train

train = my_sun(train)
#######################################################################
#'X00'~'X39'데이터만 필요하다
X_train = train.loc[:,'X00':'X39']

# standardization을 위해 평균과 표준편차 구하기
MEAN = X_train.mean()
STD = X_train.std()

# 표준편차가 0일 경우 대비하여 1e-07 추가
X_train = (X_train - MEAN) / (STD + 1e-07)

#Y데이터로는 Y14,Y15,Y16을 사용한다.
y_train = train.loc[:,'Y14':'Y16']


#시계열 분석을 위해 형태를 바꿔준다
def convert_to_timeseries(df, interval):
    sequence_list = []
    target_list = []

    for i in tqdm(range(df.shape[0] - interval)):
        sequence_list.append(np.array(df.iloc[i:i + interval, :-1]))
        target_list.append(df.iloc[i + interval, -1])

    sequence = np.array(sequence_list)
    target = np.array(target_list)
    return sequence, target

X_train_sequence= np.empty((0, 12, 32))
y_train_sequence= np.empty((0,))

for column in y_train:
    concat = pd.concat([X_train, train[column]], axis=1)
    _sequence, _target = convert_to_timeseries(concat.head(144*30), interval=12)
    X_train_sequence= np.vstack((X_train_sequence, _sequence))
    y_train_sequence= np.hstack((y_train_sequence, _target))

#3일동안의 X데이터를 같은 형태로 만들어주기 위한 과정
X_train['dummy'] = 0
X_val_sequence, _ = convert_to_timeseries(X_train, interval=12)
X_val_sequence = X_val_sequence[-432:, :, :]
X_train.drop('dummy', axis = 1, inplace = True)




#콜백에 원하는 함수를 정의한다
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 4:
            print('\n Loss is under 4, stop training')
            self.model.stop_training = True

k=5
num_val_samples = len(X_train_sequence) //k
all_score=[]

for i in range(k):
    print('working fold #', i)
    val_data = X_train_sequence[i * num_val_samples:(i+1) * num_val_samples]
    val_targets = y_train_sequence[i*num_val_samples : (i+1) * num_val_samples]

    partial_train_data = np.concatenate([X_train_sequence[:i*num_val_samples],
                                         X_train_sequence[(i+1)*num_val_samples : ]], axis = 0)
    partial_train_target = np.concatenate([y_train_sequence[:i * num_val_samples],
                                           y_train_sequence[(i + 1) * num_val_samples:]],axis=0)


    model = Sequential()
    model.add(GRU(units=64,input_shape=X_train_sequence.shape[-2:]))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse',metrics=['mae'], optimizer='adam')
    model.summary()
    callbacks = myCallback()


    model.fit(partial_train_data, partial_train_target,epochs=20,batch_size=128,
                        validation_data=(val_data, val_targets), verbose=2, shuffle=False, callbacks = [callbacks])
    final_pred = model.predict(X_val_sequence)



    val_mse,val_mae = model.evaluate(val_data, val_targets, verbose=2)
    all_score.append(val_mae)


plt.rcParams["figure.figsize"] = (15,5)
plt.plot(final_pred)
plt.show()

# 제출 파일 만들기
print(all_score)
print("%.2f%% (+/- %.2f%%)" % (np.mean(all_score), np.std(all_score)))