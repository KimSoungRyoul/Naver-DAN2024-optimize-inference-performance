
from sample_pd_to_np import preprocess_pandas
from sample_pd_to_np2 import preprocess_numpy
from line_profiler_pycharm import profile


@profile
def main():
    preprocess_pandas()

    preprocess_numpy()



if __name__ == '__main__':
    main()





