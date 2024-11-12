import pandas as pd
import numpy as np
from line_profiler_pycharm import profile
from numpy.core.records import recarray
from sklearn.preprocessing import LabelEncoder
from numpy import typing as tnp


@profile
def main():
    # 데이터 생성
    data = {
        'userid': [101, 102, 103, 104, 105],
        'search_keyword': ['생일선물', '강남맛집', '데이트코스', '분당과외', '남자패션'], # feature 설명: 검색어
        'age': [24, 27, 22, 32, 29],
        'gender': ['F', 'M', 'M', 'U', 'F'], # M:남자 F:여자 U:알수없음
        'lastlogin_at': ['2023-01-01', '2022-05-15', '2021-11-23', '2023-07-09', '2022-12-30'],
        'current_datetime': [np.datetime64("2024-10-28T12:20:32"), np.datetime64("2024-10-28T12:20:32"), # feature 설명: 지금이 몇시인지에따라 클릭할 광고가 달라질 수 있다.
                             np.datetime64("2024-10-28T12:20:32"), np.datetime64("2024-10-28T12:20:32"),
                             np.datetime64("2024-10-28T12:20:32"), ],
        'purchase_amount': [120.5, 340.75, 89.9, 560.25, 230.0],
        'is_premium': [True, True, False, True, False], # 네이버 맴버십 여부
        'click_cnt_history': [45, 70, 12, 90, 50],  # 과거 위 정보기반으로 특정광고를 클릭한 횟수 이력
        'impression_cnt_history': [3, 4, 1, 0, 3],  # 과거 위 정보기반으로 특정광고가 노출된 횟수 이력
    }

    VOCABULARY = {'생일선물': 2123, '강남맛집': 4567, '데이트코스': 5436, '분당과외': 7575, '남자패션': -1}
    GENDER_DICT = {'F': 0, 'M': 1}

    data = [
        # userid, search_keyword, age, gender, lastlogin_at, current_datetime, purchase_amount, is_premium, click_cnt_history, impression_cnt_history
        (101, "생일선물", 24, "F", "2023-01-01", "2024-10-28T12:20:32", 120.5, True, 45, 3),
        (102, "강남맛집", 27, "M", "2022-05-15", "2024-10-28T12:20:32", 340.75, True, 70, 4),
        (103, "데이트코스", 22, "M", "2021-11-23", "2024-10-28T12:20:32", 89.9, False, 12, 1),
        (104, "분당과외", 32, "U", "2023-07-09", "2024-10-28T12:20:32", 560.25, True, 90, 0),
        (105, "남자패션", 29, "F", "2022-12-30", "2024-10-28T12:20:32", 230.0, False, 50, 3),
    ]

    # DataFrame 생성
    df = pd.DataFrame(data,
        columns=["userid", "search_keyword", "age", "gender", "lastlogin_at", "current_datetime", "purchase_amount", "is_premium", "click_cnt_history", "impression_cnt_history"],
    )

    # (1) 문자열 -> 카테고리 숫자 변환


    df['search_keyword_num'] = df['search_keyword'].map(VOCABULARY).astype(np.int16)
    df['gender_num'] = df['gender'].map(GENDER_DICT).astype(np.int8)

    print("search_keyword_num:\n", df['search_keyword_num'])
    print("gender_num:\n", df['gender_num'])

    # (2) 현재 시각의 시간(hour) 추출 [NO GIL]
    df['current_hour'] = df['current_datetime'].dt.hour
    print("current_hour:\n", df['current_hour'])

    # (3) hclick_ratio 계산 [NO GIL]
    df['hclick_ratio'] = df['click_cnt_history'] / df['impression_cnt_history']
    print("hclick_ratio:\n", df['hclick_ratio'])

    # (4) 휴일 여부 계산 [NO GIL]
    df['is_business_day'] = np.is_busday(df['lastlogin_at'].values, weekmask='1111100')
    print("is_business_day:\n", df['is_business_day'])


    # recordarr = np.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
    #                          dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
    #
    # recarray.foo
    # recarray.bar

    tmp_arr = np_arr[["Purchase_Amount", "Login_Count"]]


if __name__ == '__main__':
    main()
