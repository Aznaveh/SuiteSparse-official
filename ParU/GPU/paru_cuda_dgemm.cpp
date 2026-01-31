////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_cuda_dgemm ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2025, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief      GPU-accelerated DGEMM using cuBLAS
 *
 * Performs C = -A*B + beta*C on GPU using cuBLAS
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

#ifdef PARU_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Global cuBLAS handle (created once during first use)
static cublasHandle_t paru_cublas_handle = nullptr;
static bool paru_cublas_initialized = false;

// Initialize cuBLAS handle (thread-safe via first call)
static bool paru_cuda_init_handle()
{
    if (!paru_cublas_initialized)
    {
        cublasStatus_t status = cublasCreate(&paru_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            PRLEVEL(-1, ("Error creating cuBLAS handle\n"));
            return false;
        }
        paru_cublas_initialized = true;
    }
    return true;
}

bool paru_cuda_dgemm
(
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
    if (!paru_cuda_init_handle())
    {
        PRLEVEL(-1, ("Failed to initialize cuBLAS\n"));
        return false;
    }

    // alpha is always -1 in ParU DGEMMs
    double alpha = -1.0;
    
    // Allocate device memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    cudaError_t cuda_err = cudaMalloc((void**)&d_A, M * K * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA malloc failed for d_A\n"));
        return false;
    }
    
    cuda_err = cudaMalloc((void**)&d_B, K * N * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA malloc failed for d_B\n"));
        cudaFree(d_A);
        return false;
    }
    
    cuda_err = cudaMalloc((void**)&d_C, M * N * sizeof(double));
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA malloc failed for d_C\n"));
        cudaFree(d_A);
        cudaFree(d_B);
        return false;
    }

    // Transfer data from Host to Device
    cuda_err = cudaMemcpy(d_A, A, M * K * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for A\n"));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
    cuda_err = cudaMemcpy(d_B, B, K * N * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for B\n"));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
    cuda_err = cudaMemcpy(d_C, C, M * N * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for C\n"));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    // Perform DGEMM on GPU: C = alpha*A*B + beta*C
    cublasStatus_t status = cublasDgemm(paru_cublas_handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         M, N, K,
                                         &alpha,
                                         d_A, M,
                                         d_B, K,
                                         &beta,
                                         d_C, M);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        PRLEVEL(-1, ("cuBLAS DGEMM failed\n"));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    // Transfer result back to Host
    cuda_err = cudaMemcpy(C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess)
    {
        PRLEVEL(-1, ("CUDA memcpy failed for result\n"));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return true;
}

#endif

