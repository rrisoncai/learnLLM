import numpy as np

nToken = 1000
nDim = 512
nHead = 8
d_k = nDim // nHead

X = np.random.randn(nToken, nDim)
Wq = np.random.randn(nDim, nDim)
Wk = np.random.randn(nDim, nDim)
Wv = np.random.randn(nDim, nDim)
Wo = np.random.randn(nDim, nDim)
bo = np.random.randn(1, nDim)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

out = np.zeros([nToken, nDim])
for h in range(nHead):
    ibeg = h * d_k
    iend = (h + 1) * d_k
    Wq_h = Wq[:, ibeg:iend]
    Wk_h = Wk[:, ibeg:iend]
    Wv_h = Wv[:, ibeg:iend]

    Q_h = X @ Wq_h
    K_h = X @ Wk_h
    V_h = X @ Wv_h

    scores = np.matmul(Q_h, K_h.T) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    out_h = np.matmul(attn, V_h)
    out[:, ibeg:iend] = out_h

out = out @ Wo + bo
print("output shape: ", out.shape)
