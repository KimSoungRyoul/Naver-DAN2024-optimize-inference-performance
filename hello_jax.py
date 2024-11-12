import time

import jax
import jax.numpy as jnp

# JIT 컴파일을 사용한 배열 생성 함수
@jax.jit
def create_array(n):
    return jnp.zeros((1000,1000))

# 배열 생성
s = time.process_time()
array = create_array((1000, 1000))
print((time.process_time()-s)*1000)

s = time.process_time()
array = create_array((1000, 1000))
print((time.process_time()-s)*1000)

s = time.process_time()
array = create_array((1000, 1000))
print((time.process_time()-s)*1000)

s = time.process_time()
array = create_array((1000, 1000))
print((time.process_time()-s)*1000)