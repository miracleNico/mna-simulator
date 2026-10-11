// Floating-point contraction control for the bit-exact kernels.
//
// The device kernels and the expression VM reproduce the Python core operation
// by operation, which requires every product and sum to be rounded separately
// (no fused multiply-add). GCC in ISO mode (-std=c++17) and MSVC (/fp:precise)
// do not contract by default; Clang does (-ffp-contract=on) when the target has
// FMA, e.g. with MNA_NATIVE_ARCH. MNA_NO_FP_CONTRACT at the start of a function
// body turns contraction off for that body on Clang and expands to nothing
// elsewhere.
#pragma once

#if defined(__clang__)
#define MNA_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define MNA_NO_FP_CONTRACT
#endif
