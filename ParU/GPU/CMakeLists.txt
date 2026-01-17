#-------------------------------------------------------------------------------
# PARU/GPU/CMakeLists.txt:  cmake for PARU/GPU module
#-------------------------------------------------------------------------------

# PARU/GPU Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.22 )

message ( STATUS "Building PARU_CUDA version: v"
    ${PARU_VERSION_MAJOR}.
    ${PARU_VERSION_MINOR}.
    ${PARU_VERSION_SUB} " (" ${PARU_DATE} ")" )

include ( SuiteSparsePolicy )

enable_language ( CXX )

message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )

#file ( GLOB PARU_CUDA_SOURCES "spqrgpu_*.cpp" )

if ( BUILD_SHARED_LIBS )
    target_sources ( PARU PRIVATE ${PARU_CUDA_SOURCES} )
endif ( )

if ( BUILD_STATIC_LIBS )
    target_sources ( PARU_static PRIVATE ${PARU_CUDA_SOURCES} )
endif ( )

if ( BUILD_SHARED_LIBS )
    set_target_properties ( PARU PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( PARU PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_compile_definitions ( PARU PRIVATE "PARU_HAS_CUDA" )

    target_link_libraries ( PARU PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
endif ( )

if ( BUILD_STATIC_LIBS )
    set_target_properties ( PARU_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    set_target_properties ( PARU_static PROPERTIES POSITION_INDEPENDENT_CODE ON )
    target_compile_definitions ( PARU_static PRIVATE "PARU_HAS_CUDA" )

    target_link_libraries ( PARU_static PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
endif ( )
