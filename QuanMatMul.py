import numpy as np
import time


N = 1024   # hermitian matrix dim
hermi = True  # original matrix is hermitian (True) or not (False)


def generate_H_matrix(N: int, hermi: bool = True):
    if hermi:  # original matrix is hermitian
        H = 2 * np.random.rand(N, N) - 1 + 1j * (2 * np.random.rand(N, N) - 1)
        H = (H + H.conj().T) / 2
        H -= np.eye(N) * min(np.min(np.diag(H).real), 0)
    else:  # original matrix is not hermitian
        n = N // 2
        matrix = 2 * np.random.rand(n, n) - 1 + 1j * (2 * np.random.rand(n, n) - 1)
        H = np.zeros((N, N), dtype=complex)
        H[:n, n:] = matrix
        H[n:, :n] = matrix.conj().T
    return H / np.abs(H).max()  # (N, N)

def generate_vector(N: int, hermi: bool = True):
    if hermi: # original matrix is hermitian
        return 2 * np.random.rand(N, 1) - 1 + 1j * (2 * np.random.rand(N, 1) - 1)  # (N, 1)
    else:  # original matrix is not hermitian
        vector = 2 * np.random.rand(N // 2, 1) - 1 + 1j * (2 * np.random.rand(N // 2, 1) - 1)
        return np.concatenate((np.zeros((N // 2, 1), dtype=complex), vector), axis=0)  # (N, 1)


if __name__ == '__main__':
    matrix = generate_H_matrix(N, hermi)
    vector = generate_vector(N, hermi)
    ground_truth = np.dot(matrix, vector).squeeze() / N
    if not hermi: ground_truth = ground_truth[:N // 2]
    print("ground_truth:\n", ground_truth)

    t0 = time.time()

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
    temp = (TTdag_[:, :N].transpose(1, 0, 2) * vTx).sum(axis=-1) # (N, N)
    vWTx = np.concatenate((temp, np.zeros((N, N), dtype=complex)), axis=1).ravel() # (2N^2,)

    if not hermi: # only the matrix in the upper left corner is not 0
        mT, vWTx = mT[:N * N, :N // 2], vWTx[:N * N]

    # result: x = mT^{-1} @ vWTx, square sum of residuals: resids, rank of matrix: r, singular values of matrix: s
    x, resids, r, s = np.linalg.lstsq(mT, vWTx, rcond=None)
    print("result x:\n", x)
    print("residuals:", resids)
    print("rank:", r)
    print("singular values:\n", s)
    print("error:", np.linalg.norm(ground_truth - x))

    # consumed time
    print("time:", time.time() - t0)