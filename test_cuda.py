import numpy as np
import cupy as cp
import ctypes
from sys import argv as argv
import os



def power_of_two(x):
	while x > 0:
		if x % 2 != 0:
			break
		x >>= 1
	return x == 1





def usage():
	print("USAGE: python test.py <library_file> <input_directory>") # [<output_file>]

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
	# A = bsr_matrix((Ax, Aj, Ap), shape=(M, N), blocksize=(10, 10))
	A = bsr_matrix((Ax, (Ai, Aj)), shape=(M, N), blocksize=(5, 5))
	return A, Xx, Yx, M, N, density


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



def initialize(M, N, density):
	from numpy.random import default_rng
	rng = default_rng(42)

	x = rng.random((N, ))
	y = rng.random((M, ))

	from scipy.sparse import random

	matrix = random(M, N, density=density, format='coo', dtype=np.float64, random_state=rng).asbsr(blocksize=(100, 100))

	return matrix, x, y



if __name__ == "__main__":
	# f = open("out.txt", "w")
	if len(argv) < 3:
		usage()
		exit()
	elif len(argv) > 3:
		print("Too many arguments")
		usage()
		exit()

	# threadsPerBlock = int(argv[1])
	# blocksPerGrid = int(argv[2])

	lib_file = argv[1]
	directory = argv[2]
	print(f"Testing {lib_file}")

	# Get the absolute path of the shared library
	lib_name = os.path.abspath(lib_file)

	lib = ctypes.CDLL(lib_name)

	bsr_matvec = lib.bsr_matvec

	argtypes_1 = [
		ctypes.c_int32,			   # n_brow
		ctypes.c_int32,			   # n_bcol
		ctypes.c_int32,			   # R
		ctypes.c_int32,			   # C
		ctypes.c_void_p,	 # Ap
		ctypes.c_void_p,	 # Aj
		ctypes.c_void_p,  # Ax
		ctypes.c_void_p,  # Xx
		ctypes.c_void_p,  # Yx
		ctypes.c_double,			# alpha
		ctypes.c_double,			# beta
		ctypes.c_int32,				# threadsPerBlock
		ctypes.c_int32,				# blocksPerGrid
	]

	argtypes_2 = [
		ctypes.c_int32,			   # n_brow
		ctypes.c_int32,			   # n_bcol
		ctypes.c_int32,			   # R
		ctypes.c_int32,			   # C
		ctypes.c_void_p,	 # Ap
		ctypes.c_void_p,	 # Aj
		ctypes.c_void_p,  # Ax
		ctypes.c_void_p,  # Xx
		ctypes.c_void_p,  # Yx
		ctypes.c_double,			# alpha
		ctypes.c_double,			# beta
		ctypes.c_int32,				# threadsPerBlock
		ctypes.c_int32,				# blocksPerGrid
		ctypes.c_int32,				# chunks_cnt
		ctypes.c_void_p,	 # chunk_index
		ctypes.c_void_p,	 # chunk_row
	]

	bsr_matvec.argtypes = argtypes_2

	bsr_matvec.restype = None

	# A, Xx, Yx = initialize(M, N, density)
	# A, Xx, Yx = load_from_files(M, N, density, directory="rearranged/")
	A, Xx, Yx, M, N, density = load_from_files(directory=directory)
	times = 1000
	etalon = np.fromfile(directory + f"etalon_matvec_{M}_{N}_{density}_{times}.bin", np.float64)

	nnz = A.nnz
	Ap = A.indptr
	Aj = A.indices
	Ax = A.data

	R = A.blocksize[0]
	C = A.blocksize[1]
	n_brow = M // R
	n_bcol = N // C

	print(f"{Ap[n_brow]} blocks")
	print(f"{Ap[n_brow] * R} block rows")
	print(f"{Ap[n_brow] * C} block cols")
	print(f"{Ap[n_brow] * R * C} nonzeros")

	cu_Ap = cp.array(Ap)
	cu_Aj = cp.array(Aj)
	cu_Ax = cp.array(Ax)
	cu_Xx = cp.array(Xx)
	cu_Yx = cp.array(Yx)

	alpha = np.float64(-0.7)
	beta = np.float64(0.8)

	from time import time

	cu_Yx_copy = cu_Yx.copy()


	print("Started")

	run_cnt = 1
	times = 1000
	checkpoint = 200

	Ai = np.empty((Ap[n_brow],), dtype=Ap.dtype)
	for i in range(len(Ap) - 1):
		Ai[Ap[i] : Ap[i + 1]] = i



	device = cp.cuda.Device(0)

	sm_count = device.attributes["MultiProcessorCount"]
	threadsPerSM = device.attributes["MaxThreadsPerMultiProcessor"]
	max_blocks_per_SM = device.attributes["MaxBlocksPerMultiprocessor"]

	max_blocks = max_blocks_per_SM * sm_count
	total_threads = sm_count * threadsPerSM

	REG_FILE = 1 << 16
	reg_cnt = int(input("Introduce the number of registers per thread: "))
	warp_reg_used = reg_cnt * 32
	warp_reg_reserved = (warp_reg_used // 256 + (warp_reg_used % 256 != 0)) * 256
	total_warps = REG_FILE // warp_reg_reserved
	
	print(f"Reg Used:\t{warp_reg_used}")
	print(f"Reg Reserved:\t{warp_reg_reserved}")
	print(f"Warp Count:\t{total_warps}")
	
	print("\nLaunch Configurations (threads block):")

	thread_cnt_list = []
	for warpsPerBlock in range(32, 0, -1):
		perfect_fit = total_warps % warpsPerBlock == 0
		if power_of_two(warpsPerBlock):
			threadsPerBlock = warpsPerBlock * 32
			total_blocks = total_warps // warpsPerBlock
			total_blocks = min(total_blocks, 32)

			granular_total_warps = (warpsPerBlock // 4 + (warpsPerBlock % 4 != 0)) * 4
			# print(f"{granular_total_warps = }")
			# total_reg_reserved = granular_total_warps * warp_reg_reserved
			# print(f"{total_reg_reserved = }")
			fail_by_granularity = granular_total_warps * warp_reg_reserved > REG_FILE

			print(f"\t{threadsPerBlock:4}{total_blocks:4}", end="")
			if fail_by_granularity:
				total_blocks = 0

			thread_cnt_list.append((threadsPerBlock, total_blocks * sm_count))
			if perfect_fit:
				print(" #", end="")
			print()

	best_results = []
	worst_results = []

	print(thread_cnt_list)


	# thread_cnt_list = [(threadsPerBlock, min(max_blocks, total_threads // threadsPerBlock)) for threadsPerBlock in [2**x for x in range(10, 4, -1)]]

	# thread_cnt_list += [(threadsPerBlock, total_threads // threadsPerBlock) for threadsPerBlock in [2**x for x in range(5, 3, -1)]]
	max_time = (0, 0, 0, 0.0)
	min_time = (0, 0, 0, float('inf'))

	for (threadsPerBlock, x) in thread_cnt_list:
		disqualify = False
		if x == 0:
			disqualify = True
		if threadsPerBlock == 160:
			print(f"{disqualify = }")
		min_time_crt = (threadsPerBlock, 0, 0, float('inf'))
		for oversubscription in [1, 2, 4, 8, 16]:
			if disqualify:
				break
			blocksPerGrid = x * oversubscription
			print(f"===================== {threadsPerBlock} {blocksPerGrid} {oversubscription}")

			args_normal = (
				n_brow,
				n_bcol,
				R,
				C,
				cu_Ap.data.ptr,
				cu_Aj.data.ptr,
				cu_Ax.data.ptr,
				cu_Xx.data.ptr,
				cu_Yx.data.ptr,
				alpha,
				beta,
				threadsPerBlock,
				blocksPerGrid)
			for chunks_per_thread in [1]:
				print(f"{chunks_per_thread = }")

				start_chunk_div = time()
				chunks_cnt, chunk_index, chunk_row = chunks_split(A, blocksPerGrid, chunks_per_thread)
				chunk_index_cu = cp.array(chunk_index)
				chunk_row_cu = cp.array(chunk_row)
				args_chunked = (*args_normal, chunks_cnt, chunk_index_cu.data.ptr, chunk_row_cu.data.ptr)

				# print("\n\n\n")
				# for i in range(len(args_chunked)):
				# 	print(f"{i}: {type(args_chunked[i])}")
				# print("\n\n\n")

				print(f"{chunk_index.shape = }\n\n")

				print(f"Time to split the chunks for batched version: {time() - start_chunk_div} seconds")
				average_time = 0



				for run in range(0, run_cnt):
					total_time = 0
					start = cp.cuda.Event()
					stop = cp.cuda.Event()
					start.record()
					cu_Yx[:] = cu_Yx_copy
					for crt_run in range(1, times + 1):
						# Yx = beta * Yx + alpha * A @ Xx
						bsr_matvec(*args_chunked)
						if crt_run % checkpoint == 0:
							stop.record()
							stop.synchronize()
							elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000
							print(f"{crt_run}: {elapsed}")
							total_time += elapsed
							start.record()


					print(f"{run}: {total_time}")
					# print(f"Ended {run}")
					average_time += total_time

				average_time /= run_cnt
				print(f"Average: {average_time / 5}")

				if average_time < min_time_crt[3]:
					min_time_crt = (threadsPerBlock, blocksPerGrid, oversubscription, average_time)

			Yx[:] = cu_Yx.get()
			correct = np.allclose(Yx, etalon)
			print(f"{correct = }")
			print("\n\n")
			if not correct:
				disqualify = True

		if disqualify:
			min_time_crt = (threadsPerBlock, 0, 0, float('inf'))
		best_results.append(min_time_crt)

		if (not min_time_crt[3] == float('inf')) and min_time_crt[3] > max_time[3]:
			max_time = min_time_crt
		if min_time_crt[3] < min_time[3]:
			min_time = min_time_crt


	for good_result in best_results:
		print(f"{good_result[0]} {good_result[1]} {good_result[2]} {good_result[3] / 5}")

	print()
	print(f"Best  launch: {min_time[0]} {min_time[1]} {min_time[2]} {min_time[3] / 5}")
	print(f"Worst launch: {max_time[0]} {max_time[1]} {max_time[2]} {max_time[3] / 5}")
	# f.close()

