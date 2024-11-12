from typing import Callable

import torch
import pandas as pd
import numpy as np
from line_profiler_pycharm import profile
#from numpy.core.records import recarray
#from sklearn.preprocessing import LabelEncoder
from numpy import typing as tnp
from numpy import typing as tnp
from collections import defaultdict

#  (문자열을 Int에 매핑하는) 샘플 VOCAB (못 찾으면 -1 을 반환)
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
    # userid, search_keyword, age, gender, lastlogin_at, current_datetime, purchase_amount, is_premium, click_cnt_history, impression_cnt_history
    (101, "생일선물", 24, "F", "2024-10-28", "2024-10-28T12:20:32", 120.5, True, 45, 3),
    (102, "강남맛집", 27, "M", "2024-10-28", "2024-10-28T12:20:32", 340.75, True, 70, 4),
    (103, "데이트코스", 22, "M", "2024-10-28", "2024-10-28T12:20:32", 89.9, False, 12, 1),
    (104, "분당과외", 32, "U", "2024-10-28", "2024-10-28T12:20:32", 560.25, True, 90, 0),
    (105, "남자패션", 29, "F", "2024-10-28", "2024-10-28T12:20:32", 230.0, False, 50, 3),
] * 10000


# dtype 정의 (각 열에 맞는 자료형 설정)
dtype_ = [
    ("userid", np.int32),
    ("search_keyword", "U20"),  # 최대 10글자 유니코드 문자열
    ("age", np.int8),
    ("gender", "U1"),  # 1글자 유니코드 문자열 (F, M)
    ("lastlogin_at", "M8[D]"),  # 날짜 형식
    ("current_datetime", "datetime64[h]"),  # 날짜 형식
    ("purchase_amount", np.float32),
    ("is_premium", np.bool_),
    ("impression_cnt_history", np.float32),
    ("click_cnt_history", np.float32),
]

# NumPy Structured Array
structured_np_array = np.array(data, dtype=dtype_)

# 아래와 같이 column 기준으로 데이터를 불러올 수 있습니다.
userid_arr: tnp.NDArray = structured_np_array["userid"]
search_keyword_arr: tnp.NDArray = structured_np_array["search_keyword"]
age_arr: tnp.NDArray = structured_np_array["age"]
gender_arr: tnp.NDArray = structured_np_array["gender"]
# ...
click_cnt_history_arr: tnp.NDArray = structured_np_array["click_cnt_history"]






@profile
def preprocess_numpy():
    # np array 생성
    structured_np_array = np.array(data, dtype=dtype_)

    # 아래와 같은 방식으로도 선언 가능
    @np.vectorize
    def str_to_int_vfunc(search_keyword: str)-> int:
        return VOCABULARY[search_keyword]

    # 실시간 전처리 로직

    # (1) feature1: 문자열 -> categorical num
    #  str_to_int_vfunc 함수는 tnp.NDArray 를 받아서 tnp.NDArray 를 반환하는 함수입니다.
    str_to_int_vfunc: Callable[[tnp.NDArray], tnp.NDArray] = np.vectorize(pyfunc=lambda s: VOCABULARY[s], otypes=[np.int16])

    # search_keyword: ['생일선물' '강남맛집' '데이트코스' '분당과외' '남자패션']
    search_keyword_np_arr = str_to_int_vfunc(structured_np_array["search_keyword"])
    #print("search_keyword_np_arr:",search_keyword_np_arr) # np.array([2123, 4567, 5436, 7575, -1])



    # str_gender_np_arr 샘플 데이터: ["F","M", "M", "U", "F"]
    str_gender_np_arr =  structured_np_array["gender"]

    # case 1 : np.vectorize 로 str -> int
    gender_to_int8 = np.vectorize(pyfunc=lambda s: GENDER_DICT[s], otypes=[np.int8])
    gender_np_arr_with_vector = gender_to_int8(str_gender_np_arr)
    # gender_np_arr_with_vector: [ 2  1  1  -1  2]

    # case 2: np.select 로 str -> int
    condition_list = [
        str_gender_np_arr == "U",
        str_gender_np_arr == "M",
        str_gender_np_arr == "F",
    ]
    choice_list = [-1, 1, 2]
    gender_np_arr = np.select(condition_list, choice_list)
    # gender_np_arr: [ 2  1  1 -1  2]

    print(np.equal(gender_np_arr_with_vector, gender_np_arr).all()) # True


    # (2) feature2: 날짜 자료형에서 현재 시간 추출하기   [NO GIL]
    #  ("current_datetime", "datetime64[h]"),  # 시간 단위로 날짜를 저장하는 numpy 자료형
    current_dt_np_arr = structured_np_array["current_datetime"]
    current_hour_np_arr = current_dt_np_arr.astype(np.int32) % 24

    #print("current_hour_np_arr:",current_hour_np_arr) # [12 12 12 12 12] dtype=np.int32

    # array의 dtype을 PyObject로 변환
    # pyobject_np_array = structured_np_array["click_cnt_history"].astype(np.object_) # Python float
    # pyobject_np_array2 = structured_np_array["impression_cnt_history"].astype(np.object_) # Python float

    # dtype이 np.float32 인 경우와 PyObject(np.object_)  np.divide() 연산 속도 비교



    hclick_np_arr =  structured_np_array["click_cnt_history"]
    himp_np_arr =  structured_np_array["impression_cnt_history"]
    hclick_np_arr_pyobject = hclick_np_arr.astype(object)
    himp_np_arr_pyobject = himp_np_arr.astype(object)

    # 1. numpy float32 자료형 데이터로 연산
    hclick_ratio_np_arr_np_float32 = np.divide(
        hclick_np_arr, himp_np_arr,
        dtype=np.float32
    )

    # 2. python object 자료형 데이터로 연산
    hclick_ratio_np_arr_pyobject = np.divide(
        hclick_np_arr_pyobject, himp_np_arr_pyobject,
        dtype=object,
    )


    # hclick_ratio_np_arr = np.divide(
    #     hclick_np_arr, himp_np_arr , # hclick_np_arr / himp_np_arr 연산 수행
    #     out=np.zeros_like(himp_np_arr), # where문에 만족못하는 값은 0으로 채워집니다.
    #     where=himp_np_arr != 0, # 행열의 각 요소가 이 조건에 만족하지 않으면 연산하지 않습니다.
    #     dtype=np.float32,
    # )
    # hclick_ratio_np_arr 결과값
    # np.array([0.06666667, 0.05714286, 0.08333334, 0.0, 0.06] dtype=np.float32)

    #pyobject_hclick_ratio_np_arr = np.divide(pyobject_np_array, pyobject_np_array2, dtype=np.object_)

    #print("hclick_ratio_np_arr:",hclick_ratio_np_arr) # np.array([0.06666667, 0.05714286, 0.08333334, 0.0, 0.06] dtype=np.float32)
    #print(pyobject_hclick_ratio_np_arr)

    # (4) feature4: 휴일유무 계산 [NO GIL]
    last_login_np_arr = structured_np_array["lastlogin_at"]
    # weekmask 커스텀휴일계산시 평일:1 휴일:0
    is_business_day_np_arr = np.is_busday(last_login_np_arr, weekmask="1111100")
    # is_business_day_np_arr: [ True  True  True  True  True] dtpye=np.bool_


    # array 합쳐서 전치(transposed) 시키기
    after_preprocess_np_array = np.vstack((
        structured_np_array["userid"],
        search_keyword_np_arr,
        structured_np_array["age"],
        gender_np_arr,
        is_business_day_np_arr, # lastlogin_at
        current_hour_np_arr,
        structured_np_array["purchase_amount"],
        hclick_ratio_np_arr,  # click_cnt_history / impression_cnt_history
    ),dtype=np.float32).T

    #print("after_preprocess_np_array:", after_preprocess_np_array.tolist())
    # after_preprocess_np_array: [
    #     [101.0, 2123.0, 24.0, 1.0, 12.0, 120.5, 0.06666667014360428],
    #     [102.0, 4567.0, 27.0, 1.0, 12.0, 340.75, 0.05714285746216774],
    #     [103.0, 5436.0, 22.0, 1.0, 12.0, 89.9000015258789, 0.0833333358168602],
    #     [104.0, 7575.0, 32.0, 1.0, 12.0, 560.25, 0.0],
    #     [105.0, -1.0, 29.0, 1.0, 12.0, 230.0, 0.05999999865889549]
    # ]

    # 메모리 복사가 일어나는것과 아닌것 속도 차이 비교
    input_tensor1 = torch.from_numpy(after_preprocess_np_array)


    input_tensor2 = torch.Tensor(after_preprocess_np_array)







    #print("input_tensor:", input_tensor)
    # input_tensor: tensor([[ 1.0100e+02,  2.1230e+03,  2.4000e+01,  1.0000e+00,  1.2000e+01,
    #           1.2050e+02,  6.6667e-02],
    #         [ 1.0200e+02,  4.5670e+03,  2.7000e+01,  1.0000e+00,  1.2000e+01,
    #           3.4075e+02,  5.7143e-02],
    #         [ 1.0300e+02,  5.4360e+03,  2.2000e+01,  1.0000e+00,  1.2000e+01,
    #           8.9900e+01,  8.3333e-02],
    #         [ 1.0400e+02,  7.5750e+03,  3.2000e+01,  1.0000e+00,  1.2000e+01,
    #           5.6025e+02,  0.0000e+00],
    #         [ 1.0500e+02, -1.0000e+00,  2.9000e+01,  1.0000e+00,  1.2000e+01,
    #           2.3000e+02,  6.0000e-02]])
    return input_tensor

if __name__ == '__main__':
    #preprocess_numpy()

    qqq = {"creativeCategories": [17, 70], "adAccountNo": 1309514, "campaignNo": 639036}

    print(pd.DataFrame(qqq, ))

    # data = [{"feature1": 0.333, "feature2":222}, {"feature1": 0.444, "feature2": 1111}]
    # df = pd.DataFrame(data)
    #
    # print(df)


    # pmax-cdc-ad_account
    # 	pmax-cdc-ad_account_balance
    #	pmax-cdc-ad_product_category
    # 	pmax-cdc-ad_schedule_dictionary
    # 	pmax-cdc-asset_group
    #	pmax-cdc-asset_group_signal
    # 	pmax-cdc-business_channel
    #	pmax-cdc-campaign
    # 	pmax-cdc-campaign_budget
    # 	pmax-cdc-campaign_budget_link_status
    # 	pmax-cdc-campaign_conversion_goal
    # 	pmax-cdc-campaign_criterion
    # y	pmax-cdc-location_dictionary
    # stats_campaign_daily_performance	pmax-cdc-stats_campaign_daily_performance