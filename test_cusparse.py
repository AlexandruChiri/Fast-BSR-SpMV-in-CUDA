import numpy as np
import cupy as cp
import ctypes
from sys import argv as argv
import os

def usage():
	print("USAGE: python test.py <input_directory>") # [<output_file>]

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
	A = bsr_matrix((Ax, (Ai, Aj)), shape=(M, N), blocksize=(5, 5))
	return A, Xx, Yx, M, N, density



if __name__ == "__main__":
	if len(argv) < 2:
		usage()
		exit()
	elif len(argv) > 2:
		print("Too many arguments")
		usage()
		exit()

	directory = argv[1]

	# Get the absolute path of the shared library
	lib_name = os.path.abspath("/usr/local/cuda-11.4/lib64/libcusparse.so")

	cusparse = ctypes.CDLL(lib_name)

	cusparseDbsrmv = cusparse.cusparseDbsrmv
	cusparseDbsrmv.argtypes = [
	    ctypes.c_void_p,   # handle
	    ctypes.c_int,      # dirA
	    ctypes.c_int,      # transA
	    ctypes.c_int,      # mb
	    ctypes.c_int,      # nb
	    ctypes.c_int,      # nnzb
	    ctypes.POINTER(ctypes.c_double),  # alpha
	    ctypes.c_void_p,   # descrA
	    ctypes.c_void_p,   # bsrValA (device)
	    ctypes.c_void_p,   # bsrRowPtrA (device)
	    ctypes.c_void_p,   # bsrColIndA (device)
	    ctypes.c_int,      # blockDim
	    ctypes.c_void_p,   # x (device)
	    ctypes.POINTER(ctypes.c_double),  # beta
	    ctypes.c_void_p    # y (device)
	]
	cusparseDbsrmv.restype = ctypes.c_int  # cusparseStatus_t

	bsr_matvec = cusparseDbsrmv

	cusparseCreate = cusparse.cusparseCreate
	cusparseCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
	cusparseCreate.restype = ctypes.c_int

	cusparseCreateMatDescr = cusparse.cusparseCreateMatDescr
	cusparseCreateMatDescr.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
	cusparseCreateMatDescr.restype = ctypes.c_int

	handle = ctypes.c_void_p()
	status = cusparseCreate(ctypes.byref(handle))
	if status != 0:
		raise RuntimeError(f"cusparseCreate failed with error code {status}")

	descr = ctypes.c_void_p()
	status = cusparseCreateMatDescr(ctypes.byref(descr))
	if status != 0:
		raise RuntimeError(f"cusparseCreateMatDescr failed with error code {status}")

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

	nnzb = Ap[n_brow]

	print(f"{nnzb} blocks")
	print(f"{Ap[n_brow] * R} block rows")
	print(f"{Ap[n_brow] * C} block cols")
	print(f"{Ap[n_brow] * R * C} nonzeros")

	print(f"{Ap.dtype}")
	print(f"{Aj.dtype}")

	cu_Ap = cp.array(Ap)
	cu_Aj = cp.array(Aj)
	cu_Ax = cp.array(Ax)
	cu_Xx = cp.array(Xx)
	cu_Yx = cp.array(Yx)

	alpha_np = np.float64(-1.0)
	beta_np  = np.float64(1.0)
	alpha_c = ctypes.c_double(alpha_np)
	beta_c  = ctypes.c_double(beta_np)

	from time import time

	cu_Yx_copy = cu_Yx.copy()

	average_time = 0

	print("Started")

	run_cnt = 1
	times = 1000
	checkpoint = 200

	def devptr(a): return ctypes.c_void_p(int(a.data.ptr))

	assert R == C
	assert cu_Ap.size == n_brow + 1
	assert nnzb == int(A.indptr[-1])
	assert cu_Ap.dtype == np.int32
	assert cu_Aj.dtype == np.int32
	assert cu_Ax.dtype == np.float64
	assert cu_Xx.dtype == np.float64
	assert cu_Yx.dtype == np.float64


	args = (
	    ctypes.c_void_p(handle.value),
	    ctypes.c_int(0),              # dirA = CUSPARSE_DIRECTION_ROW
	    ctypes.c_int(0),              # transA = CUSPARSE_OPERATION_NON_TRANSPOSE
	    ctypes.c_int(n_brow),
	    ctypes.c_int(n_bcol),
	    ctypes.c_int(nnzb),
	    ctypes.byref(alpha_c),
	    ctypes.c_void_p(descr.value),
	    devptr(cu_Ax),
	    devptr(cu_Ap),
	    devptr(cu_Aj),
	    ctypes.c_int(R),
	    devptr(cu_Xx),
	    ctypes.byref(beta_c),
	    devptr(cu_Yx),
	)

	device = cp.cuda.Device(0)

	sm_count = device.attributes["MultiProcessorCount"]
	threadsPerSM = device.attributes["MaxThreadsPerMultiProcessor"]
	max_blocks_per_SM = device.attributes["MaxBlocksPerMultiprocessor"]

	for run in range(0, run_cnt):
		total_time = 0
		start = cp.cuda.Event()
		stop = cp.cuda.Event()
		start.record()
		cu_Yx[:] = cu_Yx_copy
		for crt_run in range(1, times + 1):
			# Yx = beta * Yx + alpha * A @ Xx
			status = bsr_matvec(*args)
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

	Yx[:] = cu_Yx.get()
	average_time /= run_cnt
	print(f"Average: {average_time}")

	correct = np.allclose(Yx, etalon)
	print(f"{correct = }")
	print("\n\n")
