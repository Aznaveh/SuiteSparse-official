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

#ifdef PARU_USE_CUDA
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

    cublasHandle_t handle;
    cublasCreate(&handle);

    //double *d_A, *d_B, *d_C;
    //cudaMalloc(&d_A, M * K * sizeof(double));
    //cudaMalloc(&d_B, K * N * sizeof(double));
    //cudaMalloc(&d_C, M * N * sizeof(double));

    //// 4. Transfer data from Host to Device
    //cublasSetMatrix(M, K, sizeof(double), A, M, d_A, M);
    //cublasSetMatrix(K, N, sizeof(double), B, K, d_B, K);
    //cublasSetMatrix(M, N, sizeof(double), C, M, d_C, M);

    //cublasDgemm(handle,
    //            CUBLAS_OP_N, CUBLAS_OP_N,
    //            M, N, K,
    //            &alpha,
    //            d_A, M,
    //            d_B, K,
    //            &beta,
    //            d_C, M);

    //cublasGetMatrix(M, N, sizeof(double), d_C, M, C, m);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);


    return (blas_ok) ;
}
#endif

