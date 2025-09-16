import numpy as np
from scipy.sparse import coo_matrix
import random
from sys import argv as argv
from os import mkdir as mkdir


def usage():
    print("USAGE: python markov_generator.py <N> <R> <C> <density> <output_directory>")



def print_nothing(*args, **kwargs):
    None

print_aux = print


def nonzeros_cnt(data):
    cnt = 0
    for x in data:
        if not x == 0:
            cnt += 1

    return cnt


def closest_sum_list_greedy(n, target_sum):
    elements = []
    if target_sum >= n:
        elements.append(n)
        target_sum -= n

    # Greedy approach to get the desired sum
    for x in range(n-1, 0, -1):
        if target_sum == 0:
            return elements, target_sum
        elif target_sum < x:
            continue
        for _ in range(0, 2):
            if target_sum < x:
                continue
            elements.append(x)
            target_sum -= x

    return np.array(elements, dtype=np.int64), target_sum



if __name__ == '__main__':
    from numpy.random import default_rng
    rng = default_rng(42)

    if not len(argv) == 6:
        usage()
        exit()

    N = int(argv[1])
    R = int(argv[2])   # rows per block
    C = int(argv[3])   # columns per block
    density = float(argv[4])

    directory = argv[5]
    if not directory[-1] == '/':
        directory += '/'

    try:
        mkdir(directory)
        print(f"Directory \"{directory}\" created succesfully.")
    except FileExistsError:
        print(f"Directory \"{directory}\" already exists.\n")
        print("Proceeding will generate the output in it and may override existing files.")
        c = input("Do you still want to proceed? [Y/n]\n")

        while not c in 'YyNn':
            c = input(f"Unexpected string \"{c}\". Please enter \'y\' or \'n\'\n")
        if c in 'Nn':
            print("Aborted")
            exit()
        elif c in 'Yy':
            print("Proceeding")


    Xx = rng.random((N * C, )).astype(np.float64)
    Yx = rng.random((N * R, )).astype(np.float64)
    Xx[:] /= Xx.sum()
    Yx[:] /= Yx.sum()

    n_blocks = int(N * N * density)

    print = print_nothing
    print(f"{N = }; {density = }; {n_blocks = }")
    (diag_lens, sum_left) = closest_sum_list_greedy(N, n_blocks)
    print(f"Sum precision: {sum_left}")
    print(len(diag_lens))
    sign = 1
    for i in range(len(diag_lens)):
        diag_lens[i] = sign * (diag_lens[i] - N)
        sign = -sign
    print(len(diag_lens))
    blocks_on_line = np.zeros(N, dtype=np.int64)
    print(blocks_on_line)

    print_aux("1) Generated the diagonals of blocks!")

    # Keeping variable names relevant
    diag_offsets = diag_lens
    del diag_lens
    for i in diag_offsets:
        if i < 0:
            blocks_on_line[-i:] += 1
        elif i > 0:
            blocks_on_line[:-i] += 1
        else: # i == 0
            blocks_on_line[:] += 1
        print(blocks_on_line)

    print(len(blocks_on_line))

    print_aux("2) Fixed the offsets")

    nnz = n_blocks * R * C
    elems_on_line = np.zeros(N * R, dtype=blocks_on_line.dtype)
    for i in range(N):
        elems_on_line[R*i:R*(i+1)] = blocks_on_line[i] * C

    print_aux("3) Fixed the number of elemtns on each line")

    print(f"{nnz=}")
    print(f"{elems_on_line=}")

    A_rows = np.empty(nnz, dtype=np.int32)
    A_cols = np.empty(nnz, dtype=np.int32)
    A_vals = rng.random(nnz).astype(dtype=np.float64)

    # Store the offsets from which each line starts
    offsets = np.empty(N * R + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(len(elems_on_line)):
        # Obtain the offsets from where each line will start
        offsets[i+1] = offsets[i] + elems_on_line[i]
        # Ensuring the sum of the elements of each line is 1
        A_vals[offsets[i] : offsets[i+1]] /= A_vals[offsets[i] : offsets[i+1]].sum()
        # Ensuring the elements between the current offsets are on the line i
        A_rows[offsets[i] : offsets[i+1]] = i

    print_aux("4) Fixed the values and row indices for the elements")

    print(f"{offsets=}")
    print(f"{A_vals=}")
    print(f"{A_rows=}")
    print("\n\n\n\n\n")

    # Here we no longer need the offsets so we can use them as temporary variables

    # Iterate through the diagonals of blocks
    for k in diag_offsets:
        brow_start = max(0, -k)
        bcol_start = max(0, k)
        cnt = N - abs(k)
        print(f"{k=}")
        print(f"{brow_start=}")
        print(f"{bcol_start=}")
        print(f"{cnt=}")
        # Iterate through the blocks of a diagonal
        for i in range(cnt):
            brow = brow_start + i
            bcol = bcol_start + i
            print(f"\t{i=}")
            print(f"\t{brow=}")
            print(f"\t{bcol=}")
            print(f"\t{offsets[brow]=}")
            # Now going through the elements from one block
            for row in range(brow * R, (brow+1) * R):
                print(f"\t\t{row=}")
                for col in range(bcol * C, (bcol+1) * C):
                    print(f"\t\t\t{col=}")
                    print(f"\t\t\t{offsets[row]=}")
                    A_cols[offsets[row]] = col
                    offsets[row] += 1
        print(f"{offsets=}")
        print(f"{A_cols=}")
        print()

    print_aux("5) Fixed the column indices for the elements")


    # print("\n\n\n\n")
    # print(f"{A_rows=}")
    # print(f"{A_cols=}")
    # print(f"{A_vals=}")

    A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(N * R, N * C))

    del A_vals
    del A_rows
    del A_cols

    Ax = A.data

    print_aux(f"cnt = {nonzeros_cnt(Ax)}")

    print_aux("6) Generated the matrix")

    A_dense = A.toarray()

    sums = np.zeros(N * R, dtype=np.float64)
    for i in range(N * R):
        # row = A.row[i]
        # val = A.data[i]
        # sums[row] += val
        sums[i] = A_dense[i].sum()

    A_dense_flat = A_dense.ravel()
    for i in range(len(A_dense_flat)):
        A_dense_flat[i] = int(A_dense_flat[i] * 100) / 100
    del A_dense_flat

    print = print_aux

    print_aux(f"cnt = {nonzeros_cnt(A.data)}")

    # A_flat = A.data.ravel()
    # for i in range(len(A_flat)):
    #     A_flat[i] = int(A_flat[i] * 100) / 100
    # del A_flat

    # print(f"A_dense=\n{A_dense}")
    print()
    print_aux(f"cnt = {nonzeros_cnt(A.data)}")
    # print_aux(f"{A.blocksize=}")
    # print_aux(f"{A.nnz} vs {nnz}")

    # print(f"A.data=\n{A.data}")
    # print(f"A.indptr=\n{A.indptr}")

    del A_dense

    ok = True
    for i in range(N * R):
        if abs(sums[i] - 1) > 0.00000001:
            ok = False
            print_aux(f'Line {i} has the sum of {sums[i]}')
    print_aux("Tested the sum of each row")

    if not ok:
        print_aux('Not all lines have the sum of 1. Exiting...')
        exit(0)

    del sums

    print_aux('All lines have the sum of 1')
    print_aux('Outputting the matrix')
    print_aux(f'Ai: {A.row.size} {A.row.dtype}')
    print_aux(f'Aj: {A.col.size} {A.col.dtype}')
    print_aux(f'Ax: {A.data.size} {A.data.dtype}')
    print_aux(f'Xx: {Xx.size} {Xx.dtype}')
    print_aux(f'Yx: {Yx.size} {Yx.dtype}')

    A.row.tofile(directory + 'Ai')
    A.col.tofile(directory + 'Aj')
    A.data.tofile(directory + 'Ax')
    Xx.tofile(directory + 'Xx')
    Yx.tofile(directory + 'Yx')

    A.data = np.fromfile(directory + 'Ax')
    print_aux(f'cnt = {nonzeros_cnt(A.data)}')


    print_aux('7) Generating etalon')

    alpha = np.float64(-0.7)
    beta = np.float64(0.8)
    etalon_vecmat = Xx.copy()
    etalon_matvec = Yx.copy()

    times = 1000
    for _ in range(times):
        etalon_matvec[:] = beta * etalon_matvec + alpha * (A @ Xx)


    Mdim = N * R
    Ndim = N * C

    etalon_matvec.tofile(directory + f"etalon_matvec_{Mdim}_{Ndim}_{density}_{times}.bin")
    
    # for _ in range(times):
    #     etalon_vecmat[:] = beta * etalon_vecmat + alpha * (Yx @ A)
    # etalon_vecmat.tofile(directory + f"etalon_vecmat_{Mdim}_{Ndim}_{density}_{times}.bin")

    print_aux('Done')



def generate_markov_dia(n, dtype=np.float64):
    """
    Generate a sparse Markov chain transition matrix in DIA format.

    Parameters:
    - n (int): Dimension of the square matrix (n x n).
    - dtype (np.dtype): Data type of the matrix elements. Default is np.float64.

    Returns:
    - dia_matrix: Scipy sparse DIA matrix with tridiagonal structure and row sums equal to 1.
    """
    if n < 2:
        raise ValueError("Matrix size must be at least 2")

    # Initialize diagonals
    main_diag = np.full(n, 0.4, dtype=dtype)
    upper_diag = np.full(n - 1, 0.3, dtype=dtype)
    lower_diag = np.full(n - 1, 0.3, dtype=dtype)

    # First and last row adjustments to keep row sums = 1
    main_diag[0] = 0.7  # only main + upper (0.7 + 0.3 = 1)
    main_diag[-1] = 0.7  # only lower + main (0.3 + 0.7 = 1)

    # Construct the DIA matrix
    data = np.vstack([
        lower_diag,        # offset -1
        main_diag,         # offset 0
        upper_diag         # offset +1
    ])

    offsets = np.array([-1, 0, 1])

    markov_dia = dia_matrix((data, offsets), shape=(n, n))

    return markov_dia

