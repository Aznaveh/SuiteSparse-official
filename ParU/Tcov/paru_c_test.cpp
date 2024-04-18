// ==========================================================================  /
// =======================  paru_c_test.cpp =================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief    test to see how to call umfpack symbolic analysis
 *
 * @author Aznaveh
 * */
#include <math.h>

#define TEST_FREE_ALL                       \
{                                           \
    umfpack_dl_free_symbolic(&Symbolic);    \
    umfpack_dl_free_numeric(&Numeric);      \
    ParU_C_FreeNumeric(&Num, &Control);     \
    ParU_C_FreeSymbolic(&Sym, &Control);    \
    cholmod_l_free_sparse(&A, cc);          \
    cholmod_l_finish(cc);                   \
    if (B  != NULL) free(B);                \
    B  = NULL ;                             \
    if (X  != NULL) free(X);                \
    X  = NULL ;                             \
    if (b  != NULL) free(b);                \
    b  = NULL ;                             \
    if (x  != NULL) free(x);                \
    x  = NULL ;                             \
    if (xx != NULL) free(xx) ;              \
    xx = NULL ;                             \
    if (t  != NULL) free(t) ;               \
    t = NULL ;                              \
    if (T  != NULL) free(T) ;               \
    T = NULL ;                              \
}

#include "paru_cov.hpp"
extern "C"
{
#include "ParU.h"
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_C_Symbolic *Sym = NULL;
    ParU_C_Numeric *Num = NULL ;
    double *b = NULL, *B = NULL, *X = NULL, *xx = NULL, *x = NULL ;
    double *t = NULL, *T = NULL ;
    void *Symbolic = NULL, *Numeric = NULL;

    // default log10 of expected residual.  +1 means failure is expected
    double expected_log10_resid = -16 ;
    if (argc > 1)
    {
        expected_log10_resid = (double) atoi (argv [1]) ;
    }
    printf ("expected log10 of resid: %g\n", expected_log10_resid) ;

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);
    ParU_Info info;

    // null pointer tests
    info = ParU_C_Init_Control(NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    ParU_C_Control Control;
    ParU_C_Init_Control(&Control); //initialize the Control in C

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        TEST_ASSERT (expected_log10_resid == 101) ;
        return (0) ;
    }

    if (mtype != CHOLMOD_SPARSE)
    {
        TEST_ASSERT (expected_log10_resid == 102) ;
        return (0) ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int ver[3];
    char date[128];
    ParU_C_Version(ver, date);

    // Control.umfpack_ordering = UMFPACK_ORDERING_AMD;
    // Control.umfpack_strategy = UMFPACK_STRATEGY_UNSYMMETRIC;
    // Control.umfpack_strategy = UMFPACK_STRATEGY_SYMMETRIC;
    // Control.filter_singletons = 0 ;
    // Control.paru_max_threads = 6;
    Control.umfpack_ordering = UMFPACK_ORDERING_METIS_GUARD;

    // null pointer tests
    info = ParU_C_Analyze(NULL, &Sym, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Analyze(A, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Analyze(A, &Sym, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // analyze
    BRUTAL_ALLOC_TEST(info, ParU_C_Analyze(A, &Sym, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 104) ;
        return (0) ;
    }

    // null pointer tests
    info = ParU_C_Factorize(NULL, Sym, &Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Factorize(A, NULL, &Num, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Factorize(A, Sym, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Factorize(A, Sym, &Num, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // factorize
    BRUTAL_ALLOC_TEST(info, ParU_C_Factorize(A, Sym, &Num, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 105) ;
        return (0) ;
    }

    //~~~~~~~~~~~~~~~~~~~Test the results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int64_t m = Sym->m;
    // double my_time, my_solve_time;
    double resid = 0, anorm = 0, xnorm = 0;

    b = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    xx = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (xx != NULL) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;

    // null pointer tests
    info = ParU_C_Solve_Axb(NULL, Num, b, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axb(Sym, NULL, b, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axb(Sym, Num, NULL, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axb(Sym, Num, b, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axb(Sym, Num, b, xx, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // solve Ax=b
    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_Axb(Sym, Num, b, xx, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 106) ;
        return (0) ;
    }

    // compute residual for Ax=b
    BRUTAL_ALLOC_TEST(info, ParU_C_Residual_bAx(A, xx, b, &resid,
                &anorm, &xnorm, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        return (0) ;
    }
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    printf("Residual is |%.2e| anorm is %.2e, xnorm is %.2e and rcond"
            " is %.2e.\n", resid, anorm, xnorm, Num->rcond);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    // solve Ax=b using Perm, LSolve, USolve, and InvPerm:
    t = (double *) malloc (m * sizeof(double)) ;
    TEST_ASSERT (t != NULL) ;
    info = ParU_C_Perm (Num->Pfin, Num->Rs, b, m, t, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_Solve_Lxx (Sym, Num, t, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_Solve_Uxx (Sym, Num, t, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_InvPerm (Sym->Qfill, NULL, t, m, xx, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    // compute residual for Ax=b
    info = ParU_C_Residual_bAx(A, xx, b, &resid, &anorm, &xnorm, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        return (0) ;
    }
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    printf("Residual is |%.2e| anorm is %.2e, xnorm is %.2e and rcond"
            " is %.2e.\n", resid, anorm, xnorm, Num->rcond);

    // null pointer tests
    info = ParU_C_Solve_Axx(NULL, Num, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axx(Sym, NULL, b, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axx(Sym, Num, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Axx(Sym, Num, b, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Lxx (NULL, Num, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Lxx (Sym, NULL, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Lxx (Sym, Num, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Lxx (Sym, Num, t, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Uxx (NULL, Num, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Uxx (Sym, NULL, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Uxx (Sym, Num, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_Uxx (Sym, Num, t, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm (NULL, Num->Rs, b, m, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm (Num->Pfin, Num->Rs, NULL, m, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm (Num->Pfin, Num->Rs, b, m, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm (Num->Pfin, Num->Rs, b, m, t, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm (NULL, NULL, b, m, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm (Num->Pfin, NULL, NULL, m, t, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm (Num->Pfin, NULL, b, m, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm (Num->Pfin, NULL, b, m, t, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // solve Ax=b usin Axx method
    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_Axx(Sym, Num, b, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 107) ;
        return (0) ;
    }

    // compare Axx solution with Axb solution
    if (info == PARU_SUCCESS)
    {
        double err = 0 ;
        for (int64_t i = 0; i < m; ++i)
        {
            err = fmax (err, fabs (b [i] - xx [i])) ;
        }
        printf ("Axx error %g\n", err) ;
    }

    if (b  != NULL) free(b);
    b = NULL ;
    if (xx != NULL) free(xx);
    xx = NULL ;

    // construct problem for AX=B
    const int64_t nrhs = 16;  // number of right handsides
    B = (double *)malloc(m * nrhs * sizeof(double));
    TEST_ASSERT (B != NULL) ;
    X = (double *)malloc(m * nrhs * sizeof(double));
    TEST_ASSERT (X != NULL) ;
    for (int64_t i = 0; i < m; ++i)
    {
        for (int64_t j = 0; j < nrhs; ++j)
        {
            B[j * m + i] = (double)(i + j + 1);
        }
    }

    // null pointer tests
    info = ParU_C_Solve_AXB(NULL, Num, nrhs, B, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXB(Sym, NULL, nrhs, B, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXB(Sym, Num, nrhs, NULL, X, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXB(Sym, Num, nrhs, B, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXB(Sym, Num, nrhs, B, X, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // solve AX=B
    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_AXB(Sym, Num, nrhs, B, X,
                &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 109) ;
        return (0) ;
    }

    // null pointer tests
    info = ParU_C_Residual_BAX(NULL, X, B, nrhs, &resid, &anorm, &xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_BAX(A, NULL, B, nrhs, &resid, &anorm, &xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_BAX(A, X, NULL, nrhs, &resid, &anorm, &xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_BAX(A, X, B, nrhs, NULL, &anorm, &xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_BAX(A, X, B, nrhs, &resid, NULL, &xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_BAX(A, X, B, nrhs, &resid, &anorm, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_BAX(A, X, B, nrhs, &resid, &anorm, &xnorm, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // compute the residual for AX=B
    BRUTAL_ALLOC_TEST (info, ParU_C_Residual_BAX(A, X, B, nrhs, &resid,
                &anorm, &xnorm, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 110) ;
        return (0) ;
    }
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    printf("mRhs Residual is |%.2e|\n", resid);
    TEST_ASSERT (resid == 0 || log10 (resid) <= expected_log10_resid) ;

    // solve AX=B using Perm, LSolve, USolve, and InvPerm:
    T = (double *) malloc (m * nrhs * sizeof(double)) ;
    TEST_ASSERT (T != NULL) ;
    info = ParU_C_Perm_X (Num->Pfin, Num->Rs, B, m, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_Solve_LXX (Sym, Num, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_Solve_UXX (Sym, Num, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_InvPerm_X (Sym->Qfill, NULL, T, m, nrhs, X, &Control);
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;

    // compute the residual for AX=B
    info = ParU_C_Residual_BAX (A, X, B, nrhs, &resid, &anorm, &xnorm, &Control) ;
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 108) ;
        return (0) ;
    }
    resid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    printf("mRhs Residual is |%.2e| anorm is %.2e, xnorm is %.2e and rcond"
            " is %.2e.\n", resid, anorm, xnorm, Num->rcond);

    // null pointer tests
    info = ParU_C_Solve_AXX(NULL, Num, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXX(Sym, NULL, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXX(Sym, Num, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_AXX(Sym, Num, nrhs, B, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_LXX(NULL, Num, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_LXX(Sym, NULL, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_LXX(Sym, Num, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_LXX(Sym, Num, nrhs, B, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_UXX(NULL, Num, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_UXX(Sym, NULL, nrhs, B, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_UXX(Sym, Num, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Solve_UXX(Sym, Num, nrhs, B, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm_X (NULL, Num->Rs, B, m, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm_X (Num->Pfin, Num->Rs, NULL, m, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm_X (Num->Pfin, Num->Rs, B, m, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Perm_X (Num->Pfin, Num->Rs, B, m, nrhs, T, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm_X (NULL, NULL, B, m, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm_X (Num->Pfin, NULL, NULL, m, nrhs, T, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm_X (Num->Pfin, NULL, B, m, nrhs, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_InvPerm_X (Num->Pfin, NULL, B, m, nrhs, T, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // solve AX=B using AXX method
    BRUTAL_ALLOC_TEST(info, ParU_C_Solve_AXX(Sym, Num, nrhs, B, &Control));
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 111) ;
        return (0) ;
    }

    // compare AXX solution with AXB solution
    if (info == PARU_SUCCESS)
    {
        double err = 0 ;
        for (int64_t i = 0; i < m; ++i)
        {
            for (int64_t j = 0; j < nrhs; ++j)
            {
                err = fmax (err, fabs (B [i+j*m] - X [i+j*m])) ;
            }
        }
        printf ("AXX error %g\n", err) ;
    }

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int64_t max_threads = omp_get_max_threads();
    omp_set_num_threads (max_threads);

    //~~~~~~~~~~~~~~~~~~~Calling umfpack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // double umf_time = 0;

    int status ;           // Info [UMFPACK_STATUS]
    double Info[UMFPACK_INFO],  // Contains statistics about the symbolic analysis
        umf_Control[UMFPACK_CONTROL];  // it is set in umfpack_dl_defaults and
    // is used in umfpack_dl_symbolic; if
    // passed NULL it will use the defaults
    umfpack_dl_defaults(umf_Control);
    // umf_Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
    // umf_Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    // umf_Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC;
    // umf_Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
    // umf_Control [UMFPACK_SINGLETONS] = Control.filter_singletons ;
    umf_Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS_GUARD;

    int64_t *Ap = (int64_t *)A->p;
    int64_t *Ai = (int64_t *)A->i;
    double *Ax = (double *)A->x;
    // int64_t m = A->nrow;
    int64_t n = A->ncol;

    status =
        umfpack_dl_symbolic(n, n, Ap, Ai, Ax, &Symbolic, umf_Control, Info);
    // umf_Control[UMFPACK_PRL] = 0;
    // umfpack_dl_report_info(umf_Control, Info);
    TEST_ASSERT_INFO (status == UMFPACK_OK, status) ;


    status =
        umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, umf_Control, Info);
    TEST_ASSERT_INFO (status == UMFPACK_OK, status) ;

    // umf_Control[UMFPACK_PRL] = 2;
    // umfpack_dl_report_info(umf_Control, Info);
    // umfpack_dl_report_status(umf_Control, status);


    b = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (b != NULL) ;

    x = (double *)malloc(m * sizeof(double));
    TEST_ASSERT (x != NULL) ;

    for (int64_t i = 0; i < m; ++i) b[i] = i + 1;

    status = umfpack_dl_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, umf_Control,
                              Info);
    TEST_ASSERT_INFO (status == UMFPACK_OK, status) ;

    double umf_resid, umf_anorm, umf_xnorm;

    // null pointer tests
    info = ParU_C_Residual_bAx(NULL, x, b, &umf_resid, &umf_anorm, &umf_xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_bAx(A, NULL, b, &umf_resid, &umf_anorm, &umf_xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_bAx(A, x, NULL, &umf_resid, &umf_anorm, &umf_xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_bAx(A, x, b, NULL, &umf_anorm, &umf_xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_bAx(A, x, b, &umf_resid, NULL, &umf_xnorm, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_bAx(A, x, b, &umf_resid, &umf_anorm, NULL, &Control);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_Residual_bAx(A, x, b, &umf_resid, &umf_anorm, &umf_xnorm, NULL);
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    // compute the residual for Ax=b
    info = ParU_C_Residual_bAx(A, x, b, &umf_resid,
            &umf_anorm, &umf_xnorm, &Control);
    if (info != PARU_SUCCESS)
    {
        TEST_ASSERT (expected_log10_resid == 112) ;
        return (0) ;
    }
    umf_resid = (umf_anorm == 0 || umf_xnorm == 0 ) ? 0 :
        (umf_resid/(umf_anorm*umf_xnorm));
    printf("UMFPACK Residual is |%.2e| and anorm is %.2e and rcond is %.2e.\n",
           umf_resid, umf_anorm, Num->rcond);
    TEST_ASSERT (umf_resid == 0 || log10 (umf_resid) <= expected_log10_resid) ;

    // free nothing
    info = ParU_C_FreeNumeric (NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_FreeNumeric (&Num, NULL) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;
    info = ParU_C_FreeSymbolic (NULL, &Control) ;
    TEST_ASSERT_INFO (info == PARU_SUCCESS, info) ;
    info = ParU_C_FreeSymbolic (&Sym, NULL) ;
    TEST_ASSERT_INFO (info == PARU_INVALID, info) ;

    //~~~~~~~~~~~~~~~~~~~Free Everything~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEST_PASSES ;
}

