import numpy as np
import ctypes
from sys import argv as argv
import os
from time import time

def usage():
	print("USAGE: python test.py \"[<num_threads_list>]\" <library_file> <input_directory>")

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

	from scipy.sparse import bsr_matrix
	from scipy.sparse import coo_matrix
	# A = bsr_matrix((Ax, Aj, Ap), shape=(M, N), blocksize=(5, 5))
	A = bsr_matrix(coo_matrix((Ax, (Ai, Aj)), shape=(M, N)), shape=(M, N), blocksize=(5, 5))
	return A, Xx, Yx, M, N, density

def initialize(M, N, density):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))
    Y = rng.random((M, ))

    from scipy.sparse import random

    matrix = bsr_matrix(random(M, N, density=density, format='coo', dtype=np.float64, random_state=rng), blocksize=(5, 5))

    return matrix, x, y


def chunks_split(A, num_threads, chunks_per_thread):
	n_row = A.shape[0]
	n_brow = n_row // A.blocksize[0]
	nblocks = A.indptr[n_brow]
	nnz = nblocks * A.blocksize[0] * A.blocksize[1]
	blocks_per_chunk = nblocks // (num_threads * chunks_per_thread)
	elems_per_chunk = nnz // (num_threads * chunks_per_thread)
	if blocks_per_chunk < 1:
		blocks_per_chunk = 1

	Ap = A.indptr
	Aj = A.indices
	Ax = A.data

	# chunks_cnt = nblocks // blocks_per_chunk
	chunks_cnt = num_threads * chunks_per_thread

	chunk_index = np.empty(chunks_cnt + 1, dtype=np.int64)
	chunk_row = np.empty(chunks_cnt + 1, dtype=np.int32)

	chunk_index[0] = 0
	chunk_row[0] = 0
	index_crt = 0

	end = int(nnz)
	start_row = 0
	stop_row = 0

	elems_per_block = A.blocksize[0] * A.blocksize[1]

	for i in range(chunks_cnt):
		stop_index = (end * (i + 1)) // chunks_cnt

		while Ap[stop_row + 1] * elems_per_block < stop_index:
			stop_row += 1

		chunk_index[i + 1] = stop_index
		chunk_row[i + 1] = stop_row

	return chunks_cnt, chunk_index, chunk_row

if __name__ == "__main__":
	if len(argv) < 4:
		usage()
		exit()
	elif len(argv) > 4:
		print("Too many arguments")
		usage()
		exit()

	exec("threads_list = list(" + argv[1] + ")")
	lib_file = argv[2]
	directory = argv[3]
	print(f"Testing {lib_file}")

	# Get the absolute path of the shared library
	lib_name = os.path.abspath(lib_file)

	lib = ctypes.CDLL(lib_name)


	seq = lib.bsr_matvec_sequential
	normal = lib.bsr_matvec_granular_row
	fair_auto_dummy = lib.bsr_matvec_fair_auto_dummy
	fair_auto = lib.bsr_matvec_fair_auto
	task = lib.bsr_matvec_task
	fair = lib.bsr_matvec_fair
	fair_unblocked = lib.bsr_matvec_fair_unblocked
	fair_batched = lib.bsr_matvec_fair_batched
	fair_batched_unblocked = lib.bsr_matvec_fair_batched_unblocked
	fair_batched_unblocked_nodivisions = lib.bsr_matvec_fair_batched_unblocked_nodivisions

	# lista = [normal, task, fair_auto, fair, fair_unblocked, fair_batched, fair_batched_unblocked]
	lista = [seq, normal, fair_auto_dummy, fair_auto, task, fair, fair_batched, fair_batched_unblocked, fair_batched_unblocked_nodivisions]

	start_sequential = 0
	start_multithread = 1
	start_chunked = 3
	start_batched = 6

	for bsr_matvec in lista[start_sequential:start_multithread]:
		bsr_matvec.argtypes = [
			ctypes.c_int32,			   # n_brow
			ctypes.c_int32,			   # n_bcol
			ctypes.c_int32,			   # R
			ctypes.c_int32,			   # C
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Ap
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Aj
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Ax
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Xx
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Yx
			ctypes.c_double,			# alpha
			ctypes.c_double,			# beta
		]
		bsr_matvec.restype = None

	for bsr_matvec in lista[start_multithread:start_chunked]:
		bsr_matvec.argtypes = [
			ctypes.c_int32,			   # n_brow
			ctypes.c_int32,			   # n_bcol
			ctypes.c_int32,			   # R
			ctypes.c_int32,			   # C
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Ap
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Aj
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Ax
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Xx
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Yx
			ctypes.c_double,			# alpha
			ctypes.c_double,			# beta
			ctypes.c_int32				# num_threads
		]
		bsr_matvec.restype = None

	for bsr_matvec in lista[start_chunked:start_batched]:
		bsr_matvec.argtypes = [
			ctypes.c_int32,			   # n_brow
			ctypes.c_int32,			   # n_bcol
			ctypes.c_int32,			   # R
			ctypes.c_int32,			   # C
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Ap
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Aj
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Ax
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Xx
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Yx
			ctypes.c_double,			# alpha
			ctypes.c_double,			# beta
			ctypes.c_int32,				# num_threads
			ctypes.c_int32				# chunks_per_thread
		]
		bsr_matvec.restype = None

	for bsr_matvec in lista[start_batched:]:
		bsr_matvec.argtypes = [
			ctypes.c_int32,			   # n_brow
			ctypes.c_int32,			   # n_bcol
			ctypes.c_int32,			   # R
			ctypes.c_int32,			   # C
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Ap
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # Aj
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Ax
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Xx
			np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # Yx
			ctypes.c_double,			# alpha
			ctypes.c_double,			# beta
			ctypes.c_int32,				# num_threads
			ctypes.c_int32,				# chunks_cnt
			np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),	 # chunk_index
			np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),	 # chunk_row
		]
		bsr_matvec.restype = None

	# bsr_matvec = fair_auto

	# A, Xx, Yx = initialize(M, N, density)
	A, Xx, Yx, M, N, density = load_from_files(directory=directory)
	# density = 0.05
	times = 1000
	etalon = np.fromfile(directory + f"etalon_matvec_{M}_{N}_{density}_{times}.bin", np.float64)
	
	R = A.blocksize[0]
	C = A.blocksize[1]
	n_brow = M // R
	n_bcol = N // C
	Ap = A.indptr
	Aj = A.indices
	Ax = A.data


	print(f"{Ap[n_brow]} blocks")

	alpha = np.float64(-0.7)
	beta = np.float64(0.8)

	Yx_copy = Yx.copy()

	print("Started", flush=True)

	run_cnt = 1
	checkpoint = 200

	args_seq = (
		n_brow,
		n_bcol,
		R,
		C,
		Ap,
		Aj,
		Ax,
		Xx,
		Yx,
		alpha,
		beta
	)
	if False:
		print("SciPy")
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
			print("SciPy should return the right result!\nEnding execution.")
			exit(0)


	for bsr_matvec in (lista[start_sequential:start_multithread]):
		print(bsr_matvec.__name__[11:], flush=True)
		average_time = 0
		for run in range(0, run_cnt):
			start = time()
			Yx[:] = Yx_copy
			tmp_start = start
			tmp_stop = start
			for crt_run in range(1, times + 1):
				# Yx = beta * Yx + alpha * (A @ Xx)
				bsr_matvec(*args_seq)
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

		average_time /= run_cnt
		print(f"Average: {average_time}")

		correct = np.allclose(Yx, etalon)
		print(f"{correct = }")
		print(flush=True)


	for num_threads in threads_list:
		print(f" - - - - - - - THREAD_CNT = {num_threads}")

		args_normal = (
			*args_seq,
			num_threads)

				# lista_args = [args_normal, args_normal, args_normal, args_normal, args_chunked, args_batched_fair, args_batched_fair]
		lista_args = [args_normal, args_normal]

		for bsr_matvec, args in list(zip(lista[start_multithread:start_chunked], lista_args)):
			print(bsr_matvec.__name__[11:], flush=True)
			average_time = 0
			for run in range(0, run_cnt):
				start = time()
				Yx[:] = Yx_copy
				tmp_start = start
				tmp_stop = start
				for crt_run in range(1, times + 1):
					# Yx = beta * Yx + alpha * (A @ Xx)
					bsr_matvec(*args)
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

			average_time /= run_cnt
			print(f"Average: {average_time}")

			correct = np.allclose(Yx, etalon)
			print(f"{correct = }")
			print(flush=True)



		for chunks_per_thread in [8]:
			args_chunked = (*args_normal, chunks_per_thread)
			print(f"{chunks_per_thread = }")

			start = time()
			args_batched_fair = (*args_normal, *(chunks_split(A, num_threads, chunks_per_thread)))
			print(f"Time to split the chunks for batched version: {time() - start} seconds")

			# print("\n\n")
			# for i in range(len(args_batched_fair)):
			# 	print(f"{i}: {type(args_batched_fair[i])}")
			# print("\n\n")

			lista_args = [args_chunked, args_chunked, args_chunked, args_batched_fair, args_batched_fair, args_batched_fair]

			for bsr_matvec, args in list(zip(lista[start_chunked:], lista_args)):
				print(bsr_matvec.__name__[11:], flush=True)
				average_time = 0
				for run in range(0, run_cnt):
					start = time()
					Yx[:] = Yx_copy
					tmp_start = start
					tmp_stop = start
					for crt_run in range(1, times + 1):
						# Yx = beta * Yx + alpha * (A @ Xx)
						bsr_matvec(*args)
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

				average_time /= run_cnt
				print(f"Average: {average_time}")

				correct = np.allclose(Yx, etalon)
				print(f"{correct = }")
				print(flush=True)
			print("\n\n")

