from sample_compiled_mypyc import compiled_list_for_loop as compiled_list_for_loop
from sample_no_compiled_mypyc import compiled_list_for_loop as no_compiled_list_for_loop
from sample_numpy import calculate_with_numpy
from line_profiler_pycharm import profile


@profile
def main():
    batch_size = 100
    for _ in range(0,100):
        no_compiled_list_for_loop(batch_size)
        compiled_list_for_loop(batch_size)
        calculate_with_numpy(batch_size)



if __name__ == '__main__':
    main()




