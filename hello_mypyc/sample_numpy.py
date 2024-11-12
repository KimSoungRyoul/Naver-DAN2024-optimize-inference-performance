import numpy as np

def calculate_with_numpy(batch_size=10):
    input_row_dict = {
        "ad_click_cnt": np.array([10, 20, 0, 30]) * batch_size,
        "ad_impression_cnt": np.array([5, 0, 15, 25]) * batch_size,
        "ad2_click_cnt": np.array([10, 20, 0, 30]) * batch_size,
        "ad2_impression_cnt": np.array([5, 0, 15, 25]) * batch_size,
    }

    # numpy 연산을 통해 벡터화된 방식으로 전환
    ad_history_cvr = np.divide(
        input_row_dict["ad_click_cnt"], input_row_dict["ad_impression_cnt"],
        out=np.zeros_like(input_row_dict["ad_click_cnt"], dtype=np.float32), where=input_row_dict["ad_impression_cnt"] != 0
    ) * 100

    ad2_history_cvr = np.divide(
        input_row_dict["ad2_click_cnt"], input_row_dict["ad2_impression_cnt"],
        out=np.zeros_like(input_row_dict["ad2_click_cnt"], dtype=np.float32), where=input_row_dict["ad2_impression_cnt"] != 0
    ) * 100

    return {
        "ad_history_cvr": ad_history_cvr,
        "ad2_history_cvr": ad2_history_cvr,
    }
#
# # 함수 호출 예시
# result = list_comprehension()
# print(result)