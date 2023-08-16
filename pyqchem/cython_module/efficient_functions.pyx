import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()
from itertools import combinations, product
DTYPE_I = np.int64
DTYPE_F = np.float64
ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t
# cache function to accelarate FCF
def cache_fcf(func):
    cache = {}

    def wrapper(cnp.ndarray[DTYPE_I_t, ndim=1] v1, int k1,cnp.ndarray[DTYPE_I_t, ndim=1] v2,int k2,
    cnp.ndarray[DTYPE_F_t, ndim=1] ompd,
    cnp.ndarray[DTYPE_F_t, ndim=2] tpmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tqmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tr,
    cnp.ndarray[DTYPE_F_t, ndim=1] rd,
    float fcf_00,):
        key = (tuple(v1), k1, tuple(v2), k2)
        if key not in cache:
            cache[key] = func(v1, k1, v2, k2, ompd, tpmo, tqmo, tr, rd, fcf_00)
        return cache[key]

    return wrapper

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cache_fcf
def evalSingleFCFpy(
    cnp.ndarray[DTYPE_I_t, ndim=1] origin_vector,
    int k_origin,
    cnp.ndarray[DTYPE_I_t, ndim=1] target_vector,
    int k_target,
    cnp.ndarray[DTYPE_F_t, ndim=1] ompd,
    cnp.ndarray[DTYPE_F_t, ndim=2] tpmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tqmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tr,
    cnp.ndarray[DTYPE_F_t, ndim=1] rd,
    float fcf_00,
):
    cdef int ksi
    cdef int theta
    if k_origin == 0 and k_target == 0:
        return fcf_00

    if k_origin == 0:
        ksi = 0
        while target_vector[ksi] == 0:
            ksi += 1
        target_vector[ksi] -= 1

        fcf = ompd[ksi] * evalSingleFCFpy(
            origin_vector,
            0,
            target_vector,
            k_target - 1,
            ompd,
            tpmo,
            tqmo,
            tr,
            rd,
            fcf_00,
        )

        if k_target > 1:
            for theta in range(ksi, len(target_vector)):
                if target_vector[theta] > 0:
                    tmp_dbl = tpmo[ksi, theta] * np.sqrt(target_vector[theta])
                    target_vector[theta] -= 1
                    tmp_dbl *= evalSingleFCFpy(
                        origin_vector,
                        0,
                        target_vector,
                        k_target - 2,
                        ompd,
                        tpmo,
                        tqmo,
                        tr,
                        rd,
                        fcf_00,
                    )
                    fcf += tmp_dbl
                    target_vector[theta] += 1
        fcf /= np.sqrt(target_vector[ksi] + 1)
        target_vector[ksi] += 1

    else:
        ksi = 0
        while origin_vector[ksi] == 0:
            ksi += 1

        origin_vector[ksi] -= 1
        fcf = -rd[ksi] * evalSingleFCFpy(
            origin_vector,
            k_origin - 1,
            target_vector,
            k_target,
            ompd,
            tpmo,
            tqmo,
            tr,
            rd,
            fcf_00,
        )

        for theta in range(ksi, len(target_vector)):
            if origin_vector[theta] > 0:
                tmp_dbl = tqmo[ksi, theta] * np.sqrt(origin_vector[theta])
                origin_vector[theta] -= 1
                tmp_dbl *= evalSingleFCFpy(
                    origin_vector,
                    k_origin - 2,
                    target_vector,
                    k_target,
                    ompd,
                    tpmo,
                    tqmo,
                    tr,
                    rd,
                    fcf_00,
                )
                fcf += tmp_dbl
                origin_vector[theta] += 1

        if k_target > 0:
            for theta in range(len(target_vector)):
                if target_vector[theta] > 0:
                    tmp_dbl = tr[ksi, theta] * np.sqrt(target_vector[theta])
                    target_vector[theta] -= 1
                    tmp_dbl *= evalSingleFCFpy(
                        origin_vector,
                        k_origin - 1,
                        target_vector,
                        k_target - 1,
                        ompd,
                        tpmo,
                        tqmo,
                        tr,
                        rd,
                        fcf_00,
                    )
                    fcf += tmp_dbl
                    target_vector[theta] += 1

        fcf /= np.sqrt(origin_vector[ksi] + 1)
        origin_vector[ksi] += 1

    return fcf

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_fc1(
    cnp.ndarray[DTYPE_I_t, ndim=1] state,
    cnp.ndarray[DTYPE_F_t, ndim=1] ompd,
    cnp.ndarray[DTYPE_F_t, ndim=2] tpmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tqmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tr,
    cnp.ndarray[DTYPE_F_t, ndim=1] rd,
    float fcf_00,
    float eps=1e-12,
    int w_max=100,
    int min_q=5,
):
    cdef int n_modes, mode, q
    cdef cnp.ndarray[DTYPE_I_t, ndim=1] vector
    cdef cnp.ndarray[DTYPE_F_t, ndim=2] fc1
    cdef list c1
    n_modes = len(state)
    fc1 = np.zeros((n_modes, w_max))
    c1 = []
    for mode in range(n_modes):
        vector = np.zeros(n_modes).astype("int")
        for q in range(w_max):
            if q > min_q and np.abs(np.max(fc1[mode, q - min_q : q])) <= eps:
                break
            vector[mode] = q + 1
            fc1[mode, q] = evalSingleFCFpy(
                state, np.sum(state), vector, q + 1, ompd, tpmo, tqmo, tr, rd, fcf_00
            )
            c1.append(vector.copy())
    return fc1, c1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_fc2(
    cnp.ndarray[DTYPE_I_t, ndim=1] state,
    cnp.ndarray[DTYPE_F_t, ndim=2] fc1,
    cnp.ndarray[DTYPE_F_t, ndim=1] ompd,
    cnp.ndarray[DTYPE_F_t, ndim=2] tpmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tqmo,
    cnp.ndarray[DTYPE_F_t, ndim=2] tr,
    cnp.ndarray[DTYPE_F_t, ndim=1] rd,
    float fcf_00,
    float eps=1e-12,
    int w_max=100,
    int min_q=5,
):
    cdef int n_modes, p, k, l
    cdef cnp.ndarray[DTYPE_I_t, ndim=1] vector, vec
    cdef cnp.ndarray[DTYPE_F_t, ndim=3] fc2
    cdef list c2
    n_modes = len(state)
    fc2 = np.zeros((n_modes, n_modes, w_max))
    c2 = []
    for k, l in combinations(range(n_modes), 2):
        vector = np.zeros(n_modes).astype("int")
        for p in range(1, w_max):
            vector[k] = p
            vector[l] = p
            fcf = evalSingleFCFpy(
                state,
                np.sum(state),
                vector,
                2*p,
                ompd,
                tpmo,
                tqmo,
                tr,
                rd,
                fcf_00,
            )
            fc2[k, l, p] = fcf * fcf - fc1[k, p] * fc1[l, p] / (fcf_00 * fcf_00)
            if p > min_q and np.abs(np.max(fc2[k, l, p - min_q : p])) <= eps:
                for i in product(range(1, p+1),repeat=2):
                    vec = np.zeros(n_modes).astype("int")
                    np.put(vec, (k, l), i)
                    c2.append(vec)
                break
    return fc2, c2

def append_transitions(trans_list,
                       fcf_list,
                        t_class,
                        state,
                        ompd,
                        tpmo,
                        tqmo,
                        tr,
                        rd,
                        fcf_00, ):
            state_ex = np.sum(state)
            for c in t_class:
                fcf_list.append(
                    evalSingleFCFpy(
                        state,
                        state_ex,
                        c,
                        np.sum(c),
                        ompd,
                        tpmo,
                        tqmo,
                        tr,
                        rd,
                        fcf_00,
                    )
                )
                trans_list.append((state, c))