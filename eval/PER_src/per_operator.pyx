# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool
ctypedef np.float32_t FLOAT_t # cost type
ctypedef np.intp_t IND_t # array index type
ctypedef np.int32_t INT_t # array index type
CTYPE = np.float32 # cost type

def needleman_wunsch_align_score(seq1, seq2, d, m, r, normalize = True):

    N1, N2 = seq1.shape[0], seq2.shape[0]
    return _needleman_wunsch_align_score(N1, N2, seq1, seq2, d, m, r, normalize)

cpdef _needleman_wunsch_align_score(IND_t N1, IND_t N2, INT_t[:] seq1, INT_t[:] seq2,
                                    FLOAT_t d, FLOAT_t m, FLOAT_t r, bool normalized):

    # Fill up the errors
    cdef IND_t i, j
    cdef FLOAT_t match, v1, v2, v3, res
    cdef FLOAT_t[:,:] tmpRes_ = np.empty((N1 + 1, N2 + 1), dtype=CTYPE)

    for i in range(0, N1 + 1):
        tmpRes_[i][0] = i * d
    for j in range(0, N2 + 1):
        tmpRes_[0][j] = j * d

    for i in range(0, N1):
        for j in range(0, N2):
            match = r if seq1[i] == seq2[j] else m
            v1 = tmpRes_[i][j] + match
            v2 = tmpRes_[i + 1][j] + d
            v3 = tmpRes_[i][j + 1] + d
            tmpRes_[i + 1][j + 1] = max(v1, max(v2, v3))

    res = -tmpRes_[N1][N2]
    if normalized:
        res /= float(N1)
    return res
