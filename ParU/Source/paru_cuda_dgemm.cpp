////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_cuda_dgemm ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2025, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief      a wrapper around CUDA_DGEMM for tasked base dgemmed
 *
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

bool paru_cuda_dgemm
(
    int64_t f,
    int64_t M,
    int64_t N,
    int64_t K,
    double *A,
    int64_t lda,
    double *B,
    int64_t ldb,
    double beta,
    double *C,
    int64_t ldc,
    paru_work *Work,
    ParU_Numeric Num
)
{

    bool blas_ok = false;

    return (blas_ok) ;
}

