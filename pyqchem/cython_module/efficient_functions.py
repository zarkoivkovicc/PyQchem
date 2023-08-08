import numpy as np
import cython
from itertools import combinations


# cache function to accelarate FCF
def cache_fcf(func):
    cache = {}

    def wrapper(v1, k1, v2, k2, ompd, tpmo, tqmo, tr, rd, fcf_00):
        key = (tuple(v1), k1, tuple(v2), k2)
        if key not in cache:
            cache[key] = func(v1, k1, v2, k2, ompd, tpmo, tqmo, tr, rd, fcf_00)
        return cache[key]

    return wrapper


@cache_fcf
def evalSingleFCFpy(
    origin_vector, k_origin, target_vector, k_target, ompd, tpmo, tqmo, tr, rd, fcf_00
):
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


def get_fc1(
    state,
    ompd,
    tpmo,
    tqmo,
    tr,
    rd,
    fcf_00,
    eps=1e-12,
    w_max=[100] * 200,
    min_q=10,
):
    n_modes = len(state)
    fc1 = np.zeros((n_modes, np.max(w_max)))
    c1 = []
    for mode in range(n_modes):
        vector = np.zeros(n_modes).astype("int")
        for q in range(w_max[mode]):
            if q > min_q and np.abs(np.max(fc1[mode, q - min_q : q])) <= eps:
                break
            vector[mode] = q + 1
            fc1[mode, q] = evalSingleFCFpy(
                state, np.sum(state), vector, q + 1, ompd, tpmo, tqmo, tr, rd, fcf_00
            )
            c1.append(vector.copy())
    return fc1, c1


def get_fc2(
    state,
    fc1,
    ompd,
    tpmo,
    tqmo,
    tr,
    rd,
    fcf_00,
    eps=1e-12,
    w_max=[100] * 200,
    min_q=10,
):
    n_modes = len(state)
    fc2 = np.zeros((n_modes, n_modes, np.max(w_max)))
    c2 = []
    for k, l in combinations(range(n_modes), 2):
        vector = np.zeros(n_modes).astype("int")
        for p in range(1, w_max[k]):
            vector[k] = p
            for q in range(1, w_max[l]):
                vector[l] = q
                fcf = evalSingleFCFpy(
                    state,
                    np.sum(state),
                    vector,
                    vector[k] + vector[l],
                    ompd,
                    tpmo,
                    tqmo,
                    tr,
                    rd,
                    fcf_00,
                )
                if p == q:
                    fc2[k, l, p] = fcf * fcf - fc1[k, q] * fc1[l, q] / (fcf_00 * fcf_00)
                c2.append(vector.copy())
                if q > min_q and np.abs(np.max(fc2[k, l, q - min_q : q])) <= eps:
                    break
            if p > min_q and np.abs(np.max(fc2[k, l, p - min_q : p])) <= eps:
                break
    return fc2, c2
