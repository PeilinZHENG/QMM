import numpy as np
import time

N = 1024  # matrix dim
hermi = True  # Hermitian matrix (True) or not (False)


def generate_matrix(N: int, hermi: bool = True):
    matrix = 2 * np.random.rand(N, N) - 1 + 1j * (2 * np.random.rand(N, N) - 1)
    if hermi:  # Hermitian matrix
        matrix = (matrix + matrix.conj().T) / 2
        matrix -= np.eye(N) * min(np.min(np.diag(matrix).real), 0)
    return matrix / np.abs(matrix).max()  # (N, N)


if __name__ == '__main__':
    matrix = generate_matrix(N, hermi)  # (N, N)
    vector = 2 * np.random.rand(N, 1) - 1 + 1j * (2 * np.random.rand(N, 1) - 1)  # (N, 1)
    ground_truth = np.dot(matrix, vector).squeeze() / N  # (N,)

    print(f'matrix dim: {N}\nHermitian matrix: {hermi}')

    t0 = time.time()

    if hermi:
        # psi vectors: psiv
        psiv = np.concatenate((np.sqrt(matrix.conj()), np.sqrt(1 - np.abs(matrix))), axis=1) / np.sqrt(N)  # (N, 2N)

        # T matrix: mT
        mT = np.zeros((2 * N * N, N), dtype=complex)  # (2N^2, N)
        for i in range(N):
            mT[2 * N * i:2 * N * (i + 1), i] = psiv[i]

        # TTdag_ = 2 * mT @ mT^\dagger - mI
        TTdag_ = 2 * np.einsum('ni,nj->nij', psiv, psiv.conj()) - np.eye(2 * N)  # (N, 2N, 2N)

        # # W matrix: mW = S @ TTdag_ = S @ (2 * mT @ mT^\dagger - mI)
        # mW = np.zeros((2 * N * N, 2 * N * N), dtype=complex)
        # for i in range(N):
        #     mW[i:2 * N * (N - 1) + i + 1:2 * N, 2 * N * i:2 * N * (i + 1)] = TTdag_[i, :N]

        # vTx = mT @ vector
        vTx = psiv * vector  # (N, 2N)

        # vWTx = mW @ vTx
        temp = (TTdag_[:, :N].transpose(1, 0, 2) * vTx).sum(axis=-1)  # (N, N)
        vWTx = np.concatenate((temp, np.zeros((N, N), dtype=complex)), axis=1).ravel()  # (2N^2,)
    else:
        ground_truth /= 2
        mH = np.zeros((2 * N, 2 * N), dtype=complex)  # (2N, 2N)
        mH[:N, N:] = matrix
        mH[N:, :N] = matrix.conj().T

        # psi vectors: psiv
        psiv = np.concatenate((np.sqrt(mH.conj()), np.sqrt(1 - np.abs(mH))), axis=1) / np.sqrt(2 * N)  # (2N, 4N)

        # T matrix: mT
        mT = np.zeros((4 * N * N, N), dtype=complex)  # (4N^2, N)
        for i in range(N):
            mT[4 * N * i:4 * N * (i + 1), i] = psiv[i]

        psiv = psiv[N:]  # (N, 4N)

        # TTdag_ = 2 * mT @ mT^\dagger - mI
        TTdag_ = 2 * np.einsum('ni,nj->nij', psiv[:, :N], psiv.conj()) - np.eye(N, 4 * N)  # (N, N, 4N)

        # vTx = mT @ vector
        vTx = psiv * vector  # (N, 4N)

        # vWTx = mW @ vTx
        temp = (TTdag_.transpose(1, 0, 2) * vTx).sum(axis=-1)  # (N, N)
        vWTx = np.concatenate((np.zeros((N, N), dtype=complex), temp, np.zeros((N, 2 * N), dtype=complex)),
                              axis=1).ravel()  # (4N^2,)

    t1 = time.time()
    print("preparation time:", t1 - t0)
    print("ground_truth:\n", ground_truth)

    # result: x = mT^{-1} @ vWTx, square sum of residuals: resids, rank of matrix: r, singular values of matrix: s
    x, resids, r, s = np.linalg.lstsq(mT, vWTx, rcond=None)
    print("result x:\n", x)
    # print("residuals:", resids)
    # print("rank:", r)
    # print("singular values:\n", s)
    print("error:", np.linalg.norm(ground_truth - x))

    # consumed time
    t2 = time.time()
    print("solution time:", t2 - t1)
    print("total time:", t2 - t0)
