////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_cuda_dtrsm ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2025, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief      GPU-accelerated DTRSM using cuBLAS
 *
 * Solves L*X = alpha*B where L is lower triangular
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

#ifdef PARU_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

// External cuBLAS handle (defined in paru_cuda_dgemm.cpp)
extern cublasHandle_t paru_cublas_handle;
extern bool paru_cublas_initialized;

bool paru_cuda_dtrsm
(
    int64_t m,
    int64_t n,
    double alpha,
    double *a,
    int64_t lda,
    double *b,
    int64_t ldb,
    paru_work *Work,
    ParU_Numeric Num
)
{
    bool blas_ok = true;
    
    if (!paru_cublas_initialized)
    {
        PRLEVEL(-1, ("cuBLAS not initialized for DTRSM\n"));
        return false;
    }

    // Allocate device memory
    double *d_a = nullptr, *d_b = nullptr;
    
    cudaError_t cuda_err = cudaMalloc((void**)&d_a, m * m * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA malloc failed for d_a\n"));
        return false;
    }
    
    cuda_err = cudaMalloc((void**)&d_b, m * n * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA malloc failed for d_b\n"));
        cudaFree(d_a);
        return false;
    }

    // Transfer data from Host to Device
    cuda_err = cudaMemcpy(d_a, a, m * m * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for a\n"));
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }
    
    cuda_err = cudaMemcpy(d_b, b, m * n * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for b\n"));
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    // Perform DTRSM on GPU: solves L*X = alpha*B
    // L is lower triangular, stored in column-major format
    cublasStatus_t status = cublasDtrsm(paru_cublas_handle,
                                         CUBLAS_SIDE_LEFT,
                                         CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_N,
                                         CUBLAS_DIAG_UNIT,
                                         m, n,
                                         &alpha,
                                         d_a, m,
                                         d_b, m);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        PRLEVEL(-1, ("cuBLAS DTRSM failed\n"));
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    // Transfer result back to Host
    cuda_err = cudaMemcpy(b, d_b, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for result\n"));
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    cudaFree(d_a);
    cudaFree(d_b);

    return true;
}

#endif
