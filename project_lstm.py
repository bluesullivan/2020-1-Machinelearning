
#라이브러리 설치
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

#데이터를 가져온다
train = pd.read_csv('train.csv')

#데이터의 특성
temperature = ["X00","X07","X28","X31","X32"] #기온
localpress = ["X01","X06","X22","X27","X29"] #현지기압
speed= ["X02","X03","X18","X24","X26"] #풍속
water = ["X04","X10","X21","X36","X39"] #일일 누적강수량
press= ["X05","X08","X09","X23","X33"] #해면기압
sun = ["X11","X14","X16","X19","X34"] #일일 누적일사량
humidity= ["X12","X20","X30","X37","X38"] #습도
direction= ["X13","X15","X17","X25","X35"] #풍향

ylocal = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06',
          'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12',
          'Y13', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18']


#히트맵으로 파악한 고장난 센서 버리기
sensor = ['X14','X16','X19']
train.drop(sensor, axis=1, inplace=True)


#풍향 버리기
drop_wind = ['X13','X15','X17','X25','X35']
train.drop(drop_wind, axis=1, inplace=True)


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


#'X00'~'X39'데이터만 필요하다
X_train = train.loc[:,'X00':'X39']

# standardization을 위해 평균과 표준편차 구하기
MEAN = X_train.mean()
STD = X_train.std()

# 표준편차가 0일 경우 대비하여 1e-07 추가
X_train = (X_train - MEAN) / (STD + 1e-07)

#Y데이터로는 Y15,Y16을 사용한다.
y_train = train.loc[:,'Y15':'Y16']


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

#30일간의 X00~X39와 Y15,Y16을 원하는 np배열형태로 바꿔준다
for column in y_train:
    concat = pd.concat([X_train, train[column]], axis=1)

    _sequence, _target = convert_to_timeseries(concat.head(144*30), interval=12)

    X_train_sequence= np.vstack((X_train_sequence, _sequence))
    y_train_sequence= np.hstack((y_train_sequence, _target))
    #X_train_sequence.shape=(144*30,12,32)
    #y_train_sequence.shape=(144*30,)

#3일동안의 X데이터를 위와 같은 형태로 만들어주기 위한 과정
X_train['dummy'] = 0

X_val_sequence, _ = convert_to_timeseries(X_train, interval=12)

X_val_sequence = X_val_sequence[-144*2:, :, :]

X_train.drop('dummy', axis = 1, inplace = True)


#콜백에 원하는 함수를 정의한다
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #loss 값이 4보다 작아지면 overfitting을 방지하기 위해 학습을 중단한다.
        if logs.get('loss') < 4:
            print('\n Loss is under 4, stop training')
            self.model.stop_training = True

#k-fold교차검증
k=4
num_val_samples = len(X_train_sequence)//k
all_score=[]
for i in range(k):
    print('working fold #', i)
    #검증데이터
    val_data = X_train_sequence[i * num_val_samples:(i+1) * num_val_samples]
    val_targets = y_train_sequence[i*num_val_samples : (i+1) * num_val_samples]
    #학습데이터
    partial_train_data = np.concatenate([X_train_sequence[:i*num_val_samples],
                                         X_train_sequence[(i+1)*num_val_samples : ]], axis = 0)
    partial_train_target = np.concatenate([y_train_sequence[:i * num_val_samples],
                                           y_train_sequence[(i + 1) * num_val_samples:]],axis=0)

    #LSTM모델 구축
    lstm_model = tf.keras.models.Sequential()
    lstm_model.add(tf.keras.layers.LSTM(128, input_shape=X_train_sequence.shape[-2:]))
    lstm_model.add(tf.keras.layers.Dense(64, activation='linear'))
    lstm_model.add(tf.keras.layers.Dense(32, activation='linear'))
    lstm_model.add(tf.keras.layers.Dense(1))

    lstm_model.compile(optimizer='adam', loss='mse',metrics=['mae'])

    lstm_model.summary()

    #객체 생성
    callbacks = myCallback()

    # 모델 학습
    lstm_model.fit(
        partial_train_data, partial_train_target,
        epochs=80,
        batch_size=128,
        validation_data=(val_data, val_targets),
        verbose=2,
        shuffle=False,
        callbacks = [callbacks]
        )

    #fine tuning
    lstm_model.layers[0].trainable = False

    #예측하려는 실제 값으로 fine tuning을 해준다
    # Y18을 아는 하루 동안의 데이터를 가지고 fine tuning하여 overfitting의 가능성이 있지만 임의로 진행함
    finetune_X, finetune_y = convert_to_timeseries(
        pd.concat([X_train[144*30:144*31], train['Y18'][144*30:144*31]],axis = 1), interval=12)

    finetune_history = lstm_model.fit(
            finetune_X, finetune_y,
            epochs=8,
            batch_size=64,
            shuffle=False,
            verbose = 2)

    # 예측하기
    final_pred = lstm_model.predict(X_val_sequence)

    val_mse,val_mae = lstm_model.evaluate(val_data, val_targets, verbose=2)
    #k-fold, mae를 score로 저장
    all_score.append(val_mae)

print(all_score)
print("%.2f%% (+/- %.2f%%)" % (np.mean(all_score), np.std(all_score)))

#실제 값과 예측 값의 그래프 비교
final = pd.DataFrame({'id':range(0, 144*2),
                       'Y18':final_pred .reshape(1,-1)[0]})
a=np.empty((0,))
y_val=np.hstack((a,train['Y18'].tail(144*2)))
y_finalpred=final['Y18'].tail(144*2)

plt.rcParams["figure.figsize"] = (15,5)
plt.plot(y_val)
plt.plot(y_finalpred)

plt.show()
