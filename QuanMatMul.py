import numpy as np
import time
from scipy.optimize import minimize
import math
import cmath


# matrix dim
N = 32
normalize = False  # normailzed by each row (True) or max norm (False)


def generate_H_matrix(N: int, normalize: bool = True):
    # generate a N-d random complex matrix
    matrix = 2 * np.random.rand(N, N) - 1 + 1j * (2 * np.random.rand(N, N) - 1)
    if normalize:
        # calculate norms
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Add eps
        norms = np.maximum(norms, 1e-10)
        # normalize each row
        normalized_matrix = matrix / norms
        return (normalized_matrix + normalized_matrix.conj().T) / 2
    else: # normalized by the max norm
        matrix = (matrix + matrix.conj().T) / 2
        matrix -= np.eye(N) * min(np.min(np.diag(matrix).real), 0)
        return matrix / np.abs(matrix).max()


def diag_embed(x: np.array) -> np.array:
    idx = np.arange(x.shape[-1])
    res = np.zeros(x.shape + (x.shape[-1],), dtype=x.dtype)
    res[..., idx, idx] = x
    return res


if __name__ == '__main__':
    matrix = generate_H_matrix(N, normalize)
    vector1 = 2 * np.random.rand(N, 1) - 1 + 1j * (2 * np.random.rand(N, 1) - 1)
    ground_truth = np.dot(matrix, vector1).squeeze() / N
    print("ground_truth:\n", ground_truth)

    # print('----------------Old Version----------------')
    # t0 = time.time()
    # # 生成\ket{j}，其中j[j] = \ket{j}
    # j = np.array(np.zeros((2 * N, 2 * N, 1), dtype=complex))
    # for i in range(N):
    #     j[i][i][0] = 1
    #
    # # 初始化一个空列表来存储矩阵T、矩阵T*T^{\dagger}和矩阵T^{\dagger}(通过张量积构造)
    # T = []
    # TTdagger = []
    # Tdagger = []
    # psi = np.array(np.zeros((N, 2 * N, 1)), dtype=complex)
    #
    # # 矩阵形式为\sum_i\ket{}\bra{}，其中：
    # # T=\sum_i\ket{j_i}\ket{\psi_i}\bra{j_i}
    # # TT^{\dagger}=\sum_i\ket{\psi_i}\bra{\psi_i}
    # # T^{\dagger}=\sum_i\ket{j_i}\bra{\psi_i}\bra{j_i}
    # # 遍历每一项
    # for i in range(N):
    #     # 记录当前行矩阵非零项的个数
    #     d = 0
    #     # 构建\ket{\psi_i}=\frac{1}{sqrt{d}} \sum_i \sqrt{H_{ik}}\ket{k} + \sqrt{1- \abs{A_{ik}}\ket{k+N}
    #     for k in range(N):
    #         psi[i][k][0] = cmath.sqrt(matrix[i][k].conjugate())
    #         if matrix[i][k] != 0 and np.abs(matrix[i][k]) <= 1:
    #             psi[i, N + k, 0] = cmath.sqrt(1 - np.abs(matrix[i][k]))
    #             d += 1
    #     psi[i] = psi[i] / math.sqrt(d)
    #
    #     # 矩阵T的第i项
    #     t = np.kron(np.kron(j[i], psi[i].reshape(2 * N, 1)), j[i].reshape(1, 2 * N))
    #     tt = np.outer(np.kron(j[i], psi[i].reshape(2 * N, 1)),
    #                   np.kron(j[i].reshape(2 * N, 1), psi[i].reshape(2 * N, 1)))  # \sum_j\ket{\psi_j}\bra{\psi_j}
    #     tdagger = np.kron(j[i].reshape(2 * N, 1), np.kron(j[i].reshape(1, 2 * N), psi[i].reshape(1, 2 * N)))
    #
    #     # 将结果添加到列表中
    #     T.append(t)
    #     TTdagger.append(tt)
    #     Tdagger.append(tdagger)
    #
    # # 将列表转换为NumPy数组，以便进行进一步的操作或分析
    # T_array = np.array(T, dtype=complex)
    # TTdagger_array = np.array(TTdagger, dtype=complex)
    # Tdagger_array = np.array(Tdagger, dtype=complex)
    #
    # matrix_T = T_array[0]
    # matrix_TTdagger = TTdagger_array[0]
    # matrix_Tdagger = Tdagger_array[0]
    # for i in range(1, N):
    #     matrix_T += T_array[i]
    #     matrix_TTdagger += TTdagger_array[i]
    #     matrix_Tdagger += Tdagger_array[i]
    #
    # # 生成shift矩阵，满足S\ket{j,k} = \ket{k,j}
    # S = np.outer(np.kron(j[0], j[0]), np.kron(j[0], j[0]))
    # # 单位矩阵
    # I = np.outer(np.kron(j[0], j[0]), np.kron(j[0], j[0]))
    # for i in range(2 * N):
    #     for k in range(2 * N):
    #         S += np.outer(np.kron(j[i].reshape(2 * N, 1), j[k].reshape(2 * N, 1)),
    #                       np.kron(j[k].reshape(2 * N, 1), j[i].reshape(2 * N, 1)))
    #         I += np.outer(np.kron(j[i].reshape(2 * N, 1), j[k].reshape(2 * N, 1)),
    #                       np.kron(j[i].reshape(2 * N, 1), j[k].reshape(2 * N, 1)))
    # matrix_S = np.array(S - np.outer(np.kron(j[0], j[0]), np.kron(j[0], j[0])))
    # matrix_I = np.array(I - np.outer(np.kron(j[0], j[0]), np.kron(j[0], j[0])))
    #
    # # W= S(2TT^{\dagger}-I)
    # matrix_W = np.dot(matrix_S, 2 * matrix_TTdagger - matrix_I)
    #
    # # 生成2N维向量，前N维随机，后N维全0
    # vector0 = np.zeros(N).reshape(N, 1)
    #
    # vector = np.concatenate((vector1, vector0), axis=0)
    #
    # # 向量归一化
    # norms = np.linalg.norm(vector, axis=1)
    # epsilon = 1e-10
    # norms = np.maximum(norms, epsilon)
    # normalized_vector = vector / norms
    # # 理想结果：Hx/d
    # print("Hx/d:", np.dot(matrix, vector1) / d)
    # # 生成Tx
    # Tx = np.dot(matrix_T, vector)
    # WTx = np.dot(matrix_W, Tx).squeeze()
    # # 实际结果：x=T^{-1}WTx
    # x, residuals, rank, s = np.linalg.lstsq(matrix_T, WTx, rcond=None)  # x 是方程最小二乘解，residuals 是残差平方和，rank 是矩阵秩，s 是矩阵奇异值。
    # print("解向量x:\n", x)
    # error = np.linalg.norm(np.dot(matrix, vector1) / d - x[:N])
    # print("error", error)
    # t1 = time.time()
    # print("time:", t1 - t0)

    print('----------------New Version----------------')
    t0 = time.time()

    # psi vectors: psiv
    psiv = np.concatenate((np.sqrt(matrix.conj()), np.sqrt(1 - np.abs(matrix))), axis=1) / np.sqrt(N)  # (N, 2N)

    # T matrix: mT
    mT = np.zeros((2 * N * N, N), dtype=complex)
    for i in range(N):
        mT[2 * N * i:2 * N * (i + 1), i] = psiv[i]

    # TTdag_ = 2 * mT @ mT^\dagger - mI
    mI = np.tile(np.array([1, 0], dtype=complex), N).repeat(N).reshape(N, 2 * N)  # (2N^2,)
    TTdag_ = 2 * np.einsum('ni,nj->nij', psiv, psiv.conj()) - diag_embed(mI)  # (N, 2N, 2N)

    # # W matrix: mW = S @ TTdag_ = S @ (2 * mT @ mT^\dagger - mI)
    # mW = np.zeros((2 * N * N, 2 * N * N), dtype=complex)
    # for i in range(N):
    #     mW[i:2 * N * (N - 1) + i + 1:2 * N, 2 * N * i:2 * N * (i + 1)] = TTdag_[i, :N]

    # vTx = mT @ vector1
    vTx = psiv * vector1  # (N, 2N)

    # vWTx = mW @ vTx
    temp = (TTdag_[:, :N].transpose(1, 0, 2) * vTx).sum(axis=-1) # (N, N)
    vWTx = np.concatenate((temp, np.zeros((N, N), dtype=complex)), axis=1).ravel() # (2N^2,)

    # result: x = mT^{-1} @ vWTx, square sum of residuals: resids, rank of matrix: r, singular values of matrix: s
    x, resids, r, s = np.linalg.lstsq(mT, vWTx, rcond=None)
    print("result x:\n", x)
    print("residuals:", resids)
    print("rank:", r)
    print("singular values:\n", s)
    print("error:", np.linalg.norm(ground_truth - x))

    # # scipy optimizer, results are worse than those of lstsq
    # def f(x):
    #     x = x.reshape(2, N, 1)
    #     re = (psiv.real * x[0] - psiv.imag * x[1]).ravel()
    #     im = (psiv.real * x[1] + psiv.imag * x[0]).ravel()
    #     return ((re - vWTx.real) ** 2).sum() + ((im - vWTx.imag) ** 2).sum()
    #
    # result = minimize(f, 2 * np.random.rand(2 * N) - 1, method='BFGS', tol=1e-10)
    # x = result.x.reshape(2, N)
    # x = (x[0] + 1j * x[1]).squeeze()
    # print("result x:\n", x)
    # print("error:", np.linalg.norm(ground_truth - x))

    # consumed time
    print("time:", time.time() - t0)