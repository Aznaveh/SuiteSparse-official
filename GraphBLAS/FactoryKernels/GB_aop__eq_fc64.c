//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += A

#include "GB.h"
#include "GB_control.h"
#include "slice/GB_ek_slice.h"
#include "FactoryKernels/GB_aop__include.h"

// accum operator
#define GB_ACCUM_OP(z,x,y) z = GB_FC64_eq (x, y)
#define GB_Z_TYPE bool
#define GB_X_TYPE GxB_FC64_t
#define GB_Y_TYPE GxB_FC64_t
#define GB_DECLAREY(ywork) GxB_FC64_t ywork
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) ywork = Ax [(A_iso) ? 0 : (pA)]

// A and C matrices
#define GB_A_TYPE GxB_FC64_t
#define GB_C_TYPE bool
#define GB_DECLAREC(cwork) bool cwork
#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) cwork = ((GB_creal (Ax [A_iso ? 0 : (pA)]) != 0) || (GB_cimag (Ax [A_iso ? 0 : (pA)]) != 0))
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork) Cx [pC] = (A_iso) ? cwork : ((GB_creal (Ax [pA]) != 0) || (GB_cimag (Ax [pA]) != 0))
#define GB_COPY_scalar_to_C(Cx,pC,cwork) Cx [pC] = cwork
#define GB_AX_MASK(Ax,pA,asize) GB_MCAST (Ax, pA, sizeof (GxB_FC64_t))

// C(i,j) += ywork
#define GB_ACCUMULATE_scalar(Cx,pC,ywork) \
    GB_ACCUM_OP (Cx [pC], Cx [pC], ywork)

// C(i,j) += (ytype) A(i,j)
#define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork)      \
{                                                       \
    if (A_iso)                                          \
    {                                                   \
        GB_ACCUMULATE_scalar (Cx, pC, ywork) ;          \
    }                                                   \
    else                                                \
    {                                                   \
        /* A and Y have the same type here */           \
        GB_ACCUMULATE_scalar (Cx, pC, Ax [pA]) ;        \
    }                                                   \
}

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_EQ) || defined(GxB_NO_FC64) || defined(GxB_NO_EQ_FC64))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "assign/include/GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_23__eq_fc64)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    GB_Werk Werk
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += y, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_22__eq_fc64)
(
    GrB_Matrix C,
    const GB_void *ywork_handle
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    return (GrB_SUCCESS) ;
    #endif
}

