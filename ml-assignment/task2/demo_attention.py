import numpy as np
from attention import scaled_dot_product_attention

def main():
    # small example
    Q = np.array([[1., 0., 0.],
                  [0., 1., 0.]])

    K = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [1., 1., 0.]])

    V = np.array([[1., 0.],
                  [10., 0.],
                  [100., 1.]])

    # no mask
    out1, w1 = scaled_dot_product_attention(Q, K, V)
    print("Output (no mask):\n", out1)
    print("\nWeights (no mask):\n", w1)

    # mask last key for first query
    mask = np.array([[1,1,0],
                     [1,1,1]])

    out2, w2 = scaled_dot_product_attention(Q, K, V, mask)
    print("\nOutput (mask):\n", out2)
    print("\nWeights (mask):\n", w2)

if __name__ == "__main__":
    main()
