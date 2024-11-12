import pandas as pd
import numpy as np
from line_profiler_pycharm import profile
#from numpy.core.records import recarray
#from sklearn.preprocessing import LabelEncoder
from numpy import typing as tnp
from collections import defaultdict
import torch

from hello_numpy.sample_pd_to_np2 import search_keyword_arr, gender_arr

VOCABULARY = defaultdict(lambda :-1,{
    "강남맛집": 4567,
    "데이트코스": 5436,
    "생일선물": 2123,
    "분당과외": 7575,
})

GENDER_DICT = defaultdict(lambda : -1, {
    "M": 1,
    "F": 2,
})

# 샘플 데이터
data = [
    # userid, search_keyword, age, gender, lastlogin_at, current_datetime,
    # purchase_amount, is_premium, click_cnt_history, impression_cnt_history
    (101, "생일선물", 24, "F", "2024-10-28", "2024-10-28T12:20:32", 120.5, True, 45, 3),
    (102, "강남맛집", 27, "M", "2024-10-28", "2024-10-28T12:20:32", 340.75, True, 70, 4),
    (103, "데이트코스", 22, "M", "2024-10-28", "2024-10-28T12:20:32", 89.9, False, 12, 1),
    (104, "분당과외", 32, "U", "2024-10-28", "2024-10-28T12:20:32", 560.25, True, 90, 0),
    (105, "남자패션", 29, "F", "2024-10-28", "2024-10-28T12:20:32", 230.0, False, 50, 3),
]* 10000

df = pd.DataFrame(
    data,
    columns=[
        "userid", "search_keyword", "age", "gender", "lastlogin_at", "current_datetime",
        "purchase_amount", "is_premium", "impression_cnt_history", "click_cnt_history",
    ],
)

# DataFrame 으로 데이터 불러오기

user_id_series = df["userid"].array
search_keyword_series = df["search_keyword"]
age_series = df["age"]
gender_series = df["gender"]
# ...
click_cnt_history_series = df["click_cnt_history"]



@profile
def preprocess_pandas():
    # DataFrame 생성
    df = pd.DataFrame(data,
                      columns=["userid", "search_keyword", "age", "gender", "lastlogin_at", "current_datetime",
                               "purchase_amount", "is_premium", "click_cnt_history", "impression_cnt_history"],
                      )


    # (1) 문자열 -> 카테고리 숫자 변환
    df['search_keyword'] = df['search_keyword'].map(lambda s: VOCABULARY[s])
    df['gender'] = df['gender'].map(lambda s: GENDER_DICT[s])

    #print("search_keyword:\n", df['search_keyword'])
    #print("gender:\n", df['gender'])

    # (2) 현재 시각의 시간(hour) 추출  column 형 변환을 위해 메모리 복사됨 (pd.to_datetime())
    df['current_hour'] =  pd.to_datetime(df['current_datetime']).dt.hour
    #print("current_hour:\n", df['current_hour'])

    # (3) hclick_ratio 계산
    df['hclick_ratio'] = df['click_cnt_history'] / df['impression_cnt_history']
    #print("hclick_ratio:\n", df['hclick_ratio'])

    # (4) 휴일 여부 계산   python datetime으로 형 변환후 dayofweek 로 계산
    df['is_business_day'] = pd.to_datetime(df['lastlogin_at']).dt.dayofweek.isin([0, 1,2,3,4,5])
    df.drop(["lastlogin_at", "current_datetime"], axis=1, inplace=True)
    #print("is_business_day:\n", df['is_business_day'])

    input_tensor = torch.Tensor(df.values.tolist())
    #print("input_tensor:",input_tensor)
    # input_tensor: tensor([[ 1.0100e+02,  2.1230e+03,  2.4000e+01,  2.0000e+00,  1.2050e+02,
    #           1.0000e+00,  4.5000e+01,  3.0000e+00,  1.2000e+01,  1.5000e+01,
    #           1.0000e+00],
    #         [ 1.0200e+02,  4.5670e+03,  2.7000e+01,  1.0000e+00,  3.4075e+02,
    #           1.0000e+00,  7.0000e+01,  4.0000e+00,  1.2000e+01,  1.7500e+01,
    #           1.0000e+00],
    #         [ 1.0300e+02,  5.4360e+03,  2.2000e+01,  1.0000e+00,  8.9900e+01,
    #           0.0000e+00,  1.2000e+01,  1.0000e+00,  1.2000e+01,  1.2000e+01,
    #           1.0000e+00],
    #         [ 1.0400e+02,  7.5750e+03,  3.2000e+01, -1.0000e+00,  5.6025e+02,
    #           1.0000e+00,  9.0000e+01,  0.0000e+00,  1.2000e+01,         inf,
    #           1.0000e+00],
    #         [ 1.0500e+02, -1.0000e+00,  2.9000e+01,  2.0000e+00,  2.3000e+02,
    #           0.0000e+00,  5.0000e+01,  3.0000e+00,  1.2000e+01,  1.6667e+01,
    #           1.0000e+00]])
    return input_tensor


