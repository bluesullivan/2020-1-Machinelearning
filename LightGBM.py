## 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import random
from sklearn.model_selection import KFold
from tqdm import tqdm
import lightgbm as lgbm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")
random.seed(777)
np.random.seed(1)


#데이터 가져오기
train = pd.read_csv("train.csv", index_col=0)

#결측확인
train.isnull().sum()

#중복확인
train.duplicated().sum()


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
train[ylocal].head()

#범주형(카데고리) 데이터 처리   온도, 습도, 풍속
#"X00","X07","X28","X31","X32"
count, bin_dividers = np.histogram(train[temperature], bins=3)
bin_names = [1, 100, 10000] #저온은 1, 중온은 100, 고온은 10000

train['hp_bin00'] = pd.cut(x=train['X00'], #데이터 배열
bins=bin_dividers, # 경계 값 리스트
labels=bin_names, # bin 이름
include_lowest=True) # 첫 경계값 포함

train['hp_bin07'] = pd.cut(x=train['X07'], #데이터 배열
bins=bin_dividers, # 경계 값 리스트
labels=bin_names, # bin 이름
include_lowest=True) # 첫 경계값 포함


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


interval = 12
value = int(144/interval)
a = np.array([[i]*value for i in range(0, interval)]).reshape(-1)

train['date'] = np.array([[i]*144 for i in range(0,33)]).reshape(-1)
train['day'] = np.array([a for i in np.arange(0,33)]).reshape(-1)


# X = x00~x39 까지 값 , y = y00~y18 까지 값
X = train.iloc[:,np.where(train.columns.str.find('Y') != 0)[0]]
y = train.iloc[:,np.where(train.columns.str.find('Y') == 0)[0]]

X_train = X[:4320]
X_val = X[4320:]
y_train = y.drop('Y18', axis=1).dropna()
y_val = y['Y18'].dropna()


# 우선, 각각의 Y00~Y17을 예측합니다.

# 예측과 실제를 절대오류평균 MAE 계산
def rae(y_true, y_pred):
    residual = abs(y_true - y_pred)
    residual[residual < 1] = 0
    residual = residual**2
    return 'MAE', np.sum(residual)/len(residual), False

#5 fold crossvalidation 앙상블 및, train 모두를 사용하여 예측을 합니다.

def my_lgb(X_train, X_val, y_train, y_val, n_fold, param):
    val_answer = np.zeros((y_val.shape[0], 18))
    val_answer = pd.DataFrame(val_answer)
    val_answer.columns = y_train.columns
    #val_answer에 예측값을 적겠다.

    score_train = np.zeros((n_fold, 18))
    score_train = pd.DataFrame(score_train)
    score_train.columns = y_train.columns
    #score_train에 best_score_값들을 저장함


    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    # kfold 선언
    for col in tqdm(y_train.columns): #프로그래스바 진행척도는 y00~y17 즉 18번 반복한다.
        score = []
        y_train_temp = y_train[col]
        preds = np.zeros((X_val.shape[0],))

        # train split
        for n_fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)): # Kfold를 실행한다.
            trn_x, trn_y = X_train.iloc[trn_idx], y_train_temp.iloc[trn_idx]
            val_x, val_y = X_train.iloc[val_idx], y_train_temp.iloc[val_idx]
            # date 와 day는 y00~y17에서는 의미가 없는 값이니 삭제하도록합니다.
            trn_x.drop(['date', 'day'], axis=1, inplace=True)
            val_x.drop(['date', 'day'], axis=1, inplace=True)

            # LGBMRegressor 모델 생성
            reg = lgbm.LGBMRegressor(**param)
            reg.fit(trn_x, trn_y,
                    eval_set=[(trn_x, trn_y), (val_x, val_y)],
                    early_stopping_rounds=50, verbose=100,
                    eval_metric=lambda y_true, y_pred: [rae(y_true, y_pred)])
                    #평가지표는 실제값과 예측값의 MAE
            score.append(reg.best_score_['valid_1']['MAE'])
            print(score)
            # X_val 에는 Y00~Y17 실제값이 없다. 따라서 예측함에 있어 MAE를 구할
            v_p = reg.predict(X_val.drop(['date', 'day'], axis=1));
            v_p[v_p < 0] = 0
            preds += v_p / kf.n_splits
            print('%s fold 완료' % n_fold)
        # MAE를 저장 및 반환하여 다음코드에 사용한다.
        val_answer[col] = preds
        score_train[col] = score

    return val_answer, score_train


param1 = {'objective':'regression','n_estimators':10000, 'learning_rate':0.005, 'random_state':42,
        'early_stopping_rounds':50,'colsample_bytree':0.3}

param2 = {'objective':'regression','n_estimators':10000, 'learning_rate':0.005, 'random_state':42**2,
        'early_stopping_rounds':50,'colsample_bytree':0.3}

param3 = {'objective':'regression','n_estimators':10000, 'learning_rate':0.005, 'random_state':42**4,
        'early_stopping_rounds':50,'colsample_bytree':0.3}

param = {}
param[0] = param1
param[1] = param2
param[2] = param3

# 하이퍼 파라미터 값을 다르게 하여 3번 반복한 평균을 구한다.
val_answer_cv1,  score_train_cv1 = my_lgb(X_train, X_val, y_train, y_val,  5, param[0])
val_answer_cv2, score_train_cv2 = my_lgb(X_train, X_val, y_train, y_val,  5, param[1])
val_answer_cv3, score_train_cv3 = my_lgb(X_train, X_val, y_train, y_val,  5, param[2])

val_answer_cv = (val_answer_cv1 + val_answer_cv2 + val_answer_cv3)/3
score_train_cv = (score_train_cv1 + score_train_cv2 + score_train_cv3)/3


# Y_18을 예측하는 모델
# 전체 Y를 사용해서 y 조합을 짤 것입니다.
# 최적의 Y 조합으로 Y_18을 예측합니다.

def my_combination(X_val, y_val, submission_val, interval, cutoff, score):
    submission_val.index = X_val.index
    value = int(144 / interval)
    # X_val 을 interval 구간을 나누어 조합을 짤 것이다.
    a = np.array([[i] * value for i in range(0, interval)]).reshape(-1)
    X_val['day'] = np.array([a for i in np.arange(0, 3)]).reshape(-1)
    #위에서 구한 MAE 가 cutoff 이하인 것들만 뽑는다.
    index = score.mean()[score.mean() < cutoff].index
    submission_val = submission_val.loc[:, index]

    val_interval = pd.DataFrame({'Y18': []})
    val_interval.index.name = 'id'
    # 각 구간의 조합을 만들시에는 모든 경우의 수를 고려할 수 없기에
    # 3일치 Y18과 Y예측값의 MAE가 1이하인 Y끼리 sum해서 성능을 보고,
    # 그 다음 2이하인 것들끼리.. 3, 4, 5 이렇게 수를 늘려가며
    # 최종적으로 sum과 Y18의 MAE가 가장 작은 경우로 조합을 구성합니다.

    for k in range(0, interval):
        index = X_val[X_val['day'] == k].index
        score = []
        for i in submission_val.columns:
            # Y00~Y17중애서 MAE가 낮은 예측값과 Y18 차이를 sum
            score.append(((submission_val.loc[index, i].copy() - y_val[index].copy()) ** 2).sum() / (432 / interval))

        score_interval = []
        for j in range(0, 50):
            col = np.array(score) < (j + 1)
            length = len(np.where(col == True)[0])
            temp_score = ((submission_val.loc[index, col].sum(axis=1) / length - y_val[index]) ** 2).sum() / (
                        432 / interval)
            if temp_score == 0:
                score_interval.append(9999)
            else:
                score_interval.append(temp_score)
        # sum과 Y18 MAE가 가장적은 것을 최적의 조합이라 판단

        Min = np.argmin(score_interval)
        col = np.array(score) < Min + 1
        Min_length = len(np.where(col == True)[0])

        temp1_data = pd.DataFrame(submission_val.loc[index, col].sum(axis=1) / Min_length, columns=['Y18'])
        val_interval = val_interval.append(temp1_data)
        # Y18모형
    val_interval = val_interval.sort_index()
    print(sum((np.array(val_interval).reshape(-1) - np.array(y_val)) ** 2) / 432)
    return val_interval

# 3일치 Y 18과, 조합을 통해 만들어낸 Y18을 비교해보았습니다.
val_preds1 = np.zeros((X_val.shape[0],))
for i in [144]:
    val_interval1 = my_combination(X_val, y_val,  val_answer_cv, i, 2, score_train_cv)
    val_preds1 += np.array(val_interval1).reshape(-1) / 1

val_preds2 = np.zeros((X_val.shape[0],))
for i in [144]:
    val_interval2 = my_combination(X_val, y_val,val_answer_cv, i,  2, score_train_cv)
    val_preds2 += np.array(val_interval2).reshape(-1) / 1

val_preds3 = np.zeros((X_val.shape[0],))
for i in [8, 12]:
    val_interval3 = my_combination(X_val, y_val,  val_answer_cv, i,  1, score_train_cv)
    val_preds3 += np.array(val_interval3).reshape(-1) / 2


print(((np.array(y_val) - val_preds1)**2).sum()/432)
print(((np.array(y_val) - val_preds2)**2).sum()/432)
print(((np.array(y_val) - val_preds3)**2).sum()/432)

plt.rcParams["figure.figsize"] = (15,5)
plt.plot(np.array(y_val))
plt.plot(val_preds1)
plt.plot(val_preds2)
plt.plot(val_preds3)
plt.show()
