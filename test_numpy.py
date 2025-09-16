import numpy as np
import ctypes
from sys import argv as argv
import os
from time import time
from threadpoolctl import threadpool_limits

def usage():
	print("USAGE: python test.py <num_threads> <input_directory>") # [<output_file>]

def load_from_files(directory=""):
	Ai = np.fromfile(directory + "Ai", np.int32)
	Aj = np.fromfile(directory + "Aj", np.int32)
	Ax = np.fromfile(directory + "Ax", np.float64)
	Xx = np.fromfile(directory + "Xx", np.float64)
	Yx = np.fromfile(directory + "Yx", np.float64)

	M = len(Yx)
	N = len(Xx)
	density = len(Ax) / (M * N)

	density = int(density * 100) / 100

	from scipy.sparse import coo_matrix
	# A = bsr_matrix((Ax, Aj, Ap), shape=(M, N), blocksize=(5, 5))
	A = coo_matrix((Ax, (Ai, Aj)), shape=(M, N)).toarray()
	return A, Xx, Yx, M, N, density

def initialize(M, N, density):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))
    Y = rng.random((M, ))

    from scipy.sparse import random

    matrix = random(M, N, density=density, format='array', dtype=np.float64, random_state=rng)

    return matrix, x, y


if __name__ == "__main__":
	if len(argv) < 3:
		usage()
		exit()
	elif len(argv) > 3:
		print("Too many arguments")
		usage()
		exit()

	exec("threads_list = list(" + argv[1] + ")")
	directory = argv[2]

	# A, Xx, Yx = initialize(M, N, density)
	A, Xx, Yx, M, N, density = load_from_files(directory=directory)
	density = 0.05
	times = 1000
	etalon = np.fromfile(directory + f"etalon_matvec_{M}_{N}_{density}_{times}.bin", np.float64)

	alpha = np.float64(-1)
	beta = np.float64(1)

	Yx_copy = Yx.copy()

	print(f"{type(A)}")
	print("Started", flush=True)

	run_cnt = 1
	checkpoint = 200

	for num_threads in threads_list:
		print(f" - - - - - - - THREAD_CNT = {num_threads}")
		print("NumPy")
		with threadpool_limits(limits=num_threads, user_api="blas"):
			average_time = 0
			for run in range(0, run_cnt):
				start = time()
				Yx[:] = Yx_copy
				tmp_start = start
				tmp_stop = start
				for crt_run in range(1, times + 1):
					Yx[:] = beta * Yx + alpha * (A @ Xx)
					if crt_run % checkpoint == 0:
						tmp_stop = time()
						# print(f"{crt_run}: {tmp_stop - tmp_start}")
						print(f"{tmp_stop - tmp_start}")
						tmp_start = tmp_stop
					# exit(0)

				elapsed = time() - start
				print(f"{elapsed}")
				# print(f"Ended {run}")
				average_time += elapsed

			scipy_etalon = Yx
			correct = np.allclose(scipy_etalon, etalon)
			print(f"{correct = }\n")
			if not correct:
				print("NumPy should return the right result!\nEnding execution.")
				exit(0)
