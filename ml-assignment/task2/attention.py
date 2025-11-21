import numpy as np

def softmax(x):
    # stable softmax for last axis
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # turn all into arrays
    Q = np.array(Q, dtype=np.float32)
    K = np.array(K, dtype=np.float32)
    V = np.array(V, dtype=np.float32)

    # basic shape checks
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K last dimensions must match.")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have same number of rows.")

    # key dimension
    dk = K.shape[-1]

    # score matrix
    s = np.matmul(Q, K.T) / np.sqrt(dk)

    # mask support
    if mask is not None:
        m = np.array(mask)
        # broadcast mask if needed
        if m.shape != s.shape:
            try:
                m = np.broadcast_to(m, s.shape)
            except ValueError:
                raise ValueError("Mask shape cannot broadcast to scores.")
        # handle boolean or 0/1 mask
        m = (m == 1)
        s = np.where(m, s, -1e9)

    # attention weights
    w = softmax(s)

    # output
    out = np.matmul(w, V)
    return out, w