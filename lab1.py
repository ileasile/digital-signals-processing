import numpy as np
from mpl_backend_workaround import MPL_BACKEND_USED

def get_significant(v, n):
    return v[n] if 0 <= n < v.shape[0] else 0

def main():
    print("Matplotlib backend: {}".format(MPL_BACKEND_USED))
    a = np.array([-4/3, -7/12, -1/12], dtype=np.float64)
    b = np.array([1, -1, -2], dtype=np.float64)

    def sf(_n, _x):
        y = np.zeros(_n + 1)
        st = a.shape[0]
        for _i in range(0, _n + 1):
            for _j in range(st):
                idx = _i - _j - 1
                if idx >= 0:
                    y[_i] += a[_j] * y[idx]
                y[_i] += b[_j] * _x(_i - _j)

        return y[_n]

    char_p = np.poly1d( np.hstack([np.array([1]), -a]) )
    print("Characteristic polynomial: \n{}".format(char_p))
    poly_roots = char_p.roots
    print("Roots: {}".format(poly_roots))

    x1 = -1/2
    x2 = -1/3
    m = np.ndarray((3, 3))
    for i in range(3):
        m[i, 0] = x1 ** i
        m[i, 1] = i * x1 ** i
        m[i, 2] = x2 ** i
    t = np.array([b[0],
                  a[0]*b[0] + b[1],
                  a[0]**2 * b[0] + a[0] * b[1] + a[1] * b[0] + b[2]])

    print("Matrix and vector of linear system:")
    print(m)
    print(t)

    c_vec = np.linalg.solve(m, t)
    print("C_i: {}".format(c_vec))

    def hf(_n):
        return (c_vec[0] + c_vec[1] * _n) * x1 ** _n + c_vec[2] * x2 ** _n

    def delta_x(_n):
        return 1 if _n==0 else 0

    h_arg = 5
    print("h({}) = {} (using formula)".format(h_arg, hf(h_arg)))
    print("h({}) = {} (using filter)".format(h_arg, sf(h_arg, delta_x)))

    x_vec = np.array([3, 5, -2, -4, 4, 2, -1, -4, 2])
    def xf(_n):
        return get_significant(x_vec, _n)

    y_arg  = 1000
    print("y({}) = {}".format(y_arg, sf(y_arg, xf)))

    add_x  = .1
    def xf_mod(_n):
        if 0 <= _n < x_vec.shape[0]:
            return x_vec[_n] + add_x
        elif _n == -1 or _n == x_vec.shape[0]:
            return add_x
        return 0

    print("y'({}) = {}".format(y_arg, sf(y_arg, xf_mod)))

    h_vec = np.array([hf(i) for i in range(10)])
    print("h vector: {}".format(h_vec))

    def hrf(_n):
        return get_significant(h_vec, _n)

    def non_rec_filter(_h, _x, _i, _n):
        r = 0
        for k in range(_n):
            r += _x(k) * _h(_i - k)
        return r

    print("y''({}) = {}".format(y_arg, non_rec_filter(hrf, xf, y_arg, x_vec.shape[0])))

if __name__ == "__main__":
    main()