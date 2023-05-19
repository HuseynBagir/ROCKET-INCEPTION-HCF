import numba
import time

def add(x, y):
    return x + y

@numba.jit
def add_numba(x, y):
    return x + y

a = 1
b = 2

# Time the regular function
start = time.time()
for i in range(30000000):
    c = add(a, b)
end = time.time()
print("Without numba:", end - start)

# Time the numba-compiled function
start = time.time()
for i in range(1000000):
    c = add_numba(a, b)
end = time.time()
print("With numba:", end - start)
