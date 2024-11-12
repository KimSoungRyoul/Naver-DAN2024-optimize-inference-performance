

def compiled_list_for_loop(batch_size=10):
    input_row_dict = {
        "ad_click_cnt": [10, 20, 0, 30] * batch_size,
        "ad_impression_cnt": [5, 0, 15, 25] * batch_size,
        "ad2_click_cnt": [10, 20, 0, 30] * batch_size,
        "ad2_impression_cnt": [5, 0, 15, 25] * batch_size,
    }

    return {
        "ad_history_cvr": [
            _conv / _clk * 100 if _clk else 0.0
            for _conv, _clk in zip(input_row_dict["ad_click_cnt"], input_row_dict["ad_impression_cnt"])
        ],
        "ad2_history_cvr": [
            _conv / _clk * 100 if _clk else 0.0
            for _conv, _clk in zip(input_row_dict["ad2_click_cnt"], input_row_dict["ad2_impression_cnt"])
        ],
    }

# # 함수 호출 예시
# result = list_comprehension()
# print(result)