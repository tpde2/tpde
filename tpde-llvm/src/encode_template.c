// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if defined(__x86_64__)
  #include <immintrin.h>

  #define TARGET_V1 __attribute__((target("arch=opteron")))
  #define TARGET_V2 __attribute__((target("arch=x86-64-v2")))
  #define TARGET_V3 __attribute__((target("arch=x86-64-v3")))
  #define TARGET_V4 __attribute__((target("arch=x86-64-v4")))
#elif defined(__aarch64__)
  #include <arm_neon.h>

  // ARMv8.0 lacks atomic instructions (would generate libcalls)
  #define TARGET_V1 __attribute__((target("arch=armv8.1-a")))
#else
  #error "Unsupported architecture"
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef __int128_t i128;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef __uint128_t u128;

#if defined(__LDBL_MANT_DIG__) && __LDBL_MANT_DIG__ == 113
// IEEE-754 quad has 1 sign bit, 15 exponent bits, 113 mantissa bits
typedef long double fp128;
#elif defined(__FLOAT128__)
typedef __float128 fp128;
#else
  #error "unable to determine C type for fp128"
#endif

typedef i8 v8i8 __attribute__((vector_size(8)));
typedef u8 v8u8 __attribute__((vector_size(8)));
typedef i16 v4i16 __attribute__((vector_size(8)));
typedef u16 v4u16 __attribute__((vector_size(8)));
typedef i32 v2i32 __attribute__((vector_size(8)));
typedef u32 v2u32 __attribute__((vector_size(8)));
typedef float v2f32 __attribute__((vector_size(8)));

typedef i8 v16i8 __attribute__((vector_size(16)));
typedef u8 v16u8 __attribute__((vector_size(16)));
typedef i16 v8i16 __attribute__((vector_size(16)));
typedef u16 v8u16 __attribute__((vector_size(16)));
typedef i32 v4i32 __attribute__((vector_size(16)));
typedef u32 v4u32 __attribute__((vector_size(16)));
typedef i64 v2i64 __attribute__((vector_size(16)));
typedef u64 v2u64 __attribute__((vector_size(16)));
typedef float v4f32 __attribute__((vector_size(16)));
typedef double v2f64 __attribute__((vector_size(16)));

typedef bool v2i1 __attribute__((ext_vector_type(2)));
typedef bool v4i1 __attribute__((ext_vector_type(4)));
typedef bool v8i1 __attribute__((ext_vector_type(8)));
typedef bool v16i1 __attribute__((ext_vector_type(16)));

// clang-format off

// --------------------------
// loads
// --------------------------

u64 TARGET_V1 loadi8_zext(u8* ptr) { return *ptr; }
u64 TARGET_V1 loadi16_zext(u16* ptr) { return *ptr; }
u64 TARGET_V1 loadi32_zext(u32* ptr) { return *ptr; }
u64 TARGET_V1 loadi64(u64* ptr) { return *ptr; }
i64 TARGET_V1 loadi8_sext(i8 *ptr) { return *ptr; }
i64 TARGET_V1 loadi16_sext(i16 *ptr) { return *ptr; }
i64 TARGET_V1 loadi32_sext(i32 *ptr) { return *ptr; }

struct i24 { i8 data[3]; };
struct i40 { i8 data[5]; };
struct i48 { i8 data[6]; };
struct i56 { i8 data[7]; };

struct i24 TARGET_V1 loadi24(struct i24* ptr) { return *ptr; }
struct i40 TARGET_V1 loadi40(struct i40* ptr) { return *ptr; }
struct i48 TARGET_V1 loadi48(struct i48* ptr) { return *ptr; }
struct i56 TARGET_V1 loadi56(struct i56* ptr) { return *ptr; }

__uint128_t TARGET_V1 loadi128(__uint128_t* ptr) { return *ptr; }

float TARGET_V1 loadf32(float* ptr) { return *ptr; }
double TARGET_V1 loadf64(double* ptr) { return *ptr; }

#ifdef __x86_64__
__m128 TARGET_V1 loadv128(__m128_u* ptr) { return *ptr; }
__m256 TARGET_V3 loadv256(__m256_u* ptr) { return *ptr; }
__m512 TARGET_V4 loadv512(__m512_u* ptr) { return *ptr; }
#endif

#ifdef __aarch64__
uint64x2_t TARGET_V1 loadv128(uint64x2_t *ptr) { return *ptr; }
#endif

// llvm.load.relative intrinsic
u8* loadreli64(u8 *ptr, i64 off) { return &ptr[*(i32 *)(ptr + off)]; }

// --------------------------
// stores
// --------------------------

void TARGET_V1 storei8(u8* ptr, u8 value) { *ptr = value; }
void TARGET_V1 storei16(u16* ptr, u16 value) { *ptr = value; }
void TARGET_V1 storei32(u32* ptr, u32 value) { *ptr = value; }
void TARGET_V1 storei64(u64* ptr, u64 value) { *ptr = value; }

void TARGET_V1 storei24(struct i24* ptr, struct i24 value) { *ptr = value; }
void TARGET_V1 storei40(struct i40* ptr, struct i40 value) { *ptr = value; }
void TARGET_V1 storei48(struct i48* ptr, struct i48 value) { *ptr = value; }
void TARGET_V1 storei56(struct i56* ptr, struct i56 value) { *ptr = value; }

void TARGET_V1 storei128(__uint128_t* ptr, __uint128_t value) { *ptr = value; }

void TARGET_V1 storef32(float* ptr, float value) { *ptr = value; }
void TARGET_V1 storef64(double* ptr, double value) { *ptr = value; }

#ifdef __x86_64__
void TARGET_V1 storev128(__m128_u* ptr, __m128 value) { *ptr = value; }
void TARGET_V3 storev256(__m256_u* ptr, __m256 value) { *ptr = value; }
void TARGET_V4 storev512(__m512_u* ptr, __m512 value) { *ptr = value; }
#endif

#ifdef __aarch64__
void TARGET_V1 storev128(uint64x2_t *ptr, uint64x2_t value) { *ptr = value; }
#endif

// --------------------------
// integer arithmetic
// --------------------------

u32 TARGET_V1 addi32(u32 a, u32 b) { return (a + b); }
u32 TARGET_V1 subi32(u32 a, u32 b) { return (a - b); }
u32 TARGET_V1 muli32(u32 a, u32 b) { return (a * b); }
u32 TARGET_V1 udivi32(u32 a, u32 b) { return (a / b); }
i32 TARGET_V1 sdivi32(i32 a, i32 b) { return (a / b); }
u32 TARGET_V1 uremi32(u32 a, u32 b) { return (a % b); }
i32 TARGET_V1 sremi32(i32 a, i32 b) { return (a % b); }
u32 TARGET_V1 landi32(u32 a, u32 b) { return (a & b); }
u32 TARGET_V1 lori32(u32 a, u32 b) { return (a | b); }
u32 TARGET_V1 lxori32(u32 a, u32 b) { return (a ^ b); }
u32 TARGET_V1 shli32(u32 a, u32 b) { return (a << b); }
u32 TARGET_V1 shri32(u32 a, u32 b) { return (a >> b); }
i32 TARGET_V1 ashri32(i32 a, i32 b) { return (a >> b); }
i32 TARGET_V1 absi32(i32 a) { return (a < 0) ? -a : a; }

u64 TARGET_V1 addi64(u64 a, u64 b) { return (a + b); }
u64 TARGET_V1 subi64(u64 a, u64 b) { return (a - b); }
u64 TARGET_V1 muli64(u64 a, u64 b) { return (a * b); }
u64 TARGET_V1 udivi64(u64 a, u64 b) { return (a / b); }
i64 TARGET_V1 sdivi64(i64 a, i64 b) { return (a / b); }
u64 TARGET_V1 uremi64(u64 a, u64 b) { return (a % b); }
i64 TARGET_V1 sremi64(i64 a, i64 b) { return (a % b); }
u64 TARGET_V1 landi64(u64 a, u64 b) { return (a & b); }
u64 TARGET_V1 lori64(u64 a, u64 b) { return (a | b); }
u64 TARGET_V1 lxori64(u64 a, u64 b) { return (a ^ b); }
u64 TARGET_V1 shli64(u64 a, u64 b) { return (a << b); }
u64 TARGET_V1 shri64(u64 a, u64 b) { return (a >> b); }
i64 TARGET_V1 ashri64(i64 a, i64 b) { return (a >> b); }
i64 TARGET_V1 absi64(i64 a) { return (a < 0) ? -a : a; }

u128 TARGET_V1 addi128(u128 a, u128 b) { return (a + b); }
u128 TARGET_V1 subi128(u128 a, u128 b) { return (a - b); }
u128 TARGET_V1 muli128(u128 a, u128 b) { return (a * b); }
//u128 TARGET_V1 udivi128(u128 a, u128 b) { return (a / b); }
//i128 TARGET_V1 sdivi128(i128 a, i128 b) { return (a / b); }
//u128 TARGET_V1 uremi128(u128 a, u128 b) { return (a % b); }
//i128 TARGET_V1 sremi128(i128 a, i128 b) { return (a % b); }
u128 TARGET_V1 landi128(u128 a, u128 b) { return (a & b); }
u128 TARGET_V1 lori128(u128 a, u128 b) { return (a | b); }
u128 TARGET_V1 lxori128(u128 a, u128 b) { return (a ^ b); }
u128 TARGET_V1 shli128(u128 a, u128 b) { return (a << b); }
u128 TARGET_V1 shri128(u128 a, u128 b) { return (a >> b); }
i128 TARGET_V1 ashri128(i128 a, i128 b) { return (a >> b); }
u128 TARGET_V1 absi128(i128 a) { return a < 0 ? -(u128)a : a; }

// For better codegen when shifting by immediates
u128 TARGET_V1 shli128_lt64(u128 a, u64 amt, u64 iamt) {
    u64 lo = (u64)a << amt;
    u128 hi0 = (u64)a >> iamt; // iamt = 64-amt
    u128 hi1 = (u64)(a >> 64) << amt;
    return (hi0 | hi1) << 64 | lo;
}

u128 TARGET_V1 shli128_ge64(u128 a, u64 amt) {
    return a << (64 + (amt % 64));
}

u128 TARGET_V1 shri128_lt64(u128 a, u64 amt, u64 iamt) {
    u64 lo0 = (u64)a >> amt;
    u128 lo1 = (u64)(a >> 64) << iamt; // iamt = 64-amt
    u128 hi = (u64)(a >> 64) >> amt;
    return hi << 64 | lo0 | lo1;
}

u128 TARGET_V1 shri128_ge64(u128 a, u64 amt) {
    return a >> (64 + (amt % 64));
}

u128 TARGET_V1 ashri128_lt64(u128 a, u64 amt, u64 iamt) {
    u64 lo0 = (u64)a >> amt;
    u128 lo1 = (u64)(a >> 64) << iamt; // iamt = 64-amt
    u128 hi = (i64)(a >> 64) >> amt;
    return hi << 64 | lo0 | lo1;
}

u128 TARGET_V1 ashri128_ge64(i128 a, u64 amt) {
    return a >> (64 + (amt % 64));
}

// NB: UB if c is a multiple of the shift amount, but code seems ok.
u8 TARGET_V1 fshli8(u8 a, u8 b, u8 c) { return a << (c%8) | b >> (8 - (c%8)); }
u16 TARGET_V1 fshli16(u16 a, u16 b, u16 c) { return a << (c%16) | b >> (16 - (c%16)); }
u32 TARGET_V1 fshli32(u32 a, u32 b, u32 c) { return a << (c%32) | b >> (32 - (c%32)); }
u64 TARGET_V1 fshli64(u64 a, u64 b, u64 c) { return a << (c%64) | b >> (64 - (c%64)); }

u8 TARGET_V1 fshri8(u8 a, u8 b, u8 c) { return b >> (c%8) | a << (8 - (c%8)); }
u16 TARGET_V1 fshri16(u16 a, u16 b, u16 c) { return b >> (c%16) | a << (16 - (c%16)); }
u32 TARGET_V1 fshri32(u32 a, u32 b, u32 c) { return b >> (c%32) | a << (32 - (c%32)); }
u64 TARGET_V1 fshri64(u64 a, u64 b, u64 c) { return b >> (c%64) | a << (64 - (c%64)); }

u8 TARGET_V1 roli8(u8 a, u8 c) { return c ? (a << (c&7) | a >> (8 - (c&7))) : a; }
u16 TARGET_V1 roli16(u16 a, u16 c) { return c ? (a << (c&15) | a >> (16 - (c&15))) : a; }
u32 TARGET_V1 roli32(u32 a, u32 c) { return c ? (a << (c&31) | a >> (32 - (c&31))) : a; }
u64 TARGET_V1 roli64(u64 a, u64 c) { return c ? (a << (c&63) | a >> (64 - (c&63))) : a; }

u8 TARGET_V1 rori8(u8 a, u8 c) { return c ? (a >> (c&7) | a << (8 - (c&7))) : a; }
u16 TARGET_V1 rori16(u16 a, u16 c) { return c ? (a >> (c&15) | a << (16 - (c&15))) : a; }
u32 TARGET_V1 rori32(u32 a, u32 c) { return c ? (a >> (c&31) | a << (32 - (c&31))) : a; }
u64 TARGET_V1 rori64(u64 a, u64 c) { return c ? (a >> (c&63) | a << (64 - (c&63))) : a; }

u32 TARGET_V1 smini32(u32 a, u32 b) { return (i32)a < (i32)b ? a : b; }
u64 TARGET_V1 smini64(u64 a, u64 b) { return (i64)a < (i64)b ? a : b; }
u32 TARGET_V1 umini32(u32 a, u32 b) { return a < b ? a : b; }
u64 TARGET_V1 umini64(u64 a, u64 b) { return a < b ? a : b; }
u32 TARGET_V1 smaxi32(u32 a, u32 b) { return (i32)a > (i32)b ? a : b; }
u64 TARGET_V1 smaxi64(u64 a, u64 b) { return (i64)a > (i64)b ? a : b; }
u32 TARGET_V1 umaxi32(u32 a, u32 b) { return a > b ? a : b; }
u64 TARGET_V1 umaxi64(u64 a, u64 b) { return a > b ? a : b; }

i64 TARGET_V1 scmpi32(i32 a, i32 b) { return a < b ? -1 : a > b; }
i64 TARGET_V1 ucmpi32(u32 a, u32 b) { return a < b ? -1 : a > b; }
i64 TARGET_V1 scmpi64(i64 a, i64 b) { return a < b ? -1 : a > b; }
i64 TARGET_V1 ucmpi64(u64 a, u64 b) { return a < b ? -1 : a > b; }

u16 TARGET_V1 bswapi16(u16 a) { return __builtin_bswap16(a); }
u32 TARGET_V1 bswapi32(u32 a) { return __builtin_bswap32(a); }
u64 TARGET_V1 bswapi48(u64 a) { return __builtin_bswap64(a) >> 16; }
u64 TARGET_V1 bswapi64(u64 a) { return __builtin_bswap64(a); }
u128 TARGET_V1 bswapi128(u128 a) { return (u128)__builtin_bswap64(a) << 64 | __builtin_bswap64(a >> 64); }

u32 TARGET_V1 ctpopi32(u32 a) { return __builtin_popcount(a); }
u64 TARGET_V1 ctpopi64(u64 a) { return __builtin_popcountll(a); }

u32 TARGET_V1 cttzi32_zero_poison(u32 a) { return __builtin_ctz(a); }
u64 TARGET_V1 cttzi64_zero_poison(u64 a) { return __builtin_ctzll(a); }

u32 TARGET_V1 cttzi8(i8 a) { if ((u8)a == 0) { return 8; } else { return __builtin_ctz(a); }}
u32 TARGET_V1 cttzi16(i16 a) { if ((u16)a == 0) { return 16; } else { return __builtin_ctz(a); }}
u32 TARGET_V1 cttzi32(u32 a) { if (a == 0) { return 32; } else { return __builtin_ctz(a); }}
u64 TARGET_V1 cttzi64(u64 a) { if (a == 0) { return 64; } else { return __builtin_ctzll(a); }}


u32 TARGET_V1 ctlzi8_zero_poison(i8 a) { return __builtin_clz((u32)(u8)a) - 24; }
u32 TARGET_V1 ctlzi16_zero_poison(i16 a) { return __builtin_clz((u32)(u16)a) - 16; }
u32 TARGET_V1 ctlzi32_zero_poison(u32 a) { return __builtin_clz(a); }
u64 TARGET_V1 ctlzi64_zero_poison(u64 a) { return __builtin_clzll(a); }

u32 TARGET_V1 ctlzi8(i8 a) { if ((u8)a == 0) { return 8; } else { return __builtin_clz((u32)(u8)a) - 24; }}
u32 TARGET_V1 ctlzi16(i16 a) { if ((u16)a == 0) { return 16; } else { return __builtin_clz((u32)(u16)a) - 16; }}
u32 TARGET_V1 ctlzi32(u32 a) { if (a == 0) { return 32; } else { return __builtin_clz(a); }}
u64 TARGET_V1 ctlzi64(u64 a) { if (a == 0) { return 64; } else { return __builtin_clzll(a); }}

u32 TARGET_V1 bitreversei32(u32 a) { return __builtin_bitreverse32(a); }
u64 TARGET_V1 bitreversei64(u64 a) { return __builtin_bitreverse64(a); }

// --------------------------
// integer overflow
// --------------------------

#define RES_STRUCT(ty) struct res_##ty { ty val; u64 of; };
RES_STRUCT(i8)
RES_STRUCT(u8)
RES_STRUCT(i16)
RES_STRUCT(u16)
RES_STRUCT(i32)
RES_STRUCT(u32)
RES_STRUCT(i64)
RES_STRUCT(u64)
RES_STRUCT(i128)
RES_STRUCT(u128)
#undef RES_STRUCT

// Use regcall on x86-64 to return 128 bit integers + overflow flag in registers
#if defined(__x86_64__)
    #define OF_OP_CC __regcall
#else
    #define OF_OP_CC
#endif
#define OF_OP(ty, inv_ty, op)                                                  \
    OF_OP_CC struct res_##ty TARGET_V1 of_##op##_##ty(inv_ty a, inv_ty b) {    \
        ty    res;                                                             \
        _Bool of = __builtin_##op##_overflow((ty)a, (ty)b, &res);              \
        return (struct res_##ty){res, of};                                     \
    }

#define OF_OPS(width) \
    OF_OP(i##width, u##width, add) \
    OF_OP(i##width, u##width, sub) \
    OF_OP(i##width, u##width, mul) \
    OF_OP(u##width, i##width, add) \
    OF_OP(u##width, i##width, sub) \
    OF_OP(u##width, i##width, mul) \

OF_OPS(8)
OF_OPS(16)
OF_OPS(32)
OF_OPS(64)

// 128-bit mul-overflow is inlined on x86-64, but not on AArch64. Furthermore,
// on AArch64, there is no calling convention to return more than two registers
// (LLVM supports this, but Clang doesn't, because it follows the AAPCS ABI).
// Therefore, code these manually for AArch64.
#if defined(__x86_64__)
OF_OPS(128)
#endif

#undef OF_OPS
#undef OF_OP
#undef OF_OP_CC

// --------------------------
// saturating arithmetic
// --------------------------

u8 TARGET_V1 sat_add_u8(u8 a, u8 b) { u8 r; return __builtin_add_overflow(a, b, &r) ? UINT8_MAX : r; }
u8 TARGET_V1 sat_sub_u8(u8 a, u8 b) { u8 r; return __builtin_sub_overflow(a, b, &r) ? 0 : r; }
i8 TARGET_V1 sat_add_i8(i8 a, i8 b) { i8 r; return __builtin_add_overflow(a, b, &r) ? a < 0 ? INT8_MIN : INT8_MAX : r; }
i8 TARGET_V1 sat_sub_i8(i8 a, i8 b) { i8 r; return __builtin_sub_overflow(a, b, &r) ? a < 0 ? INT8_MIN : INT8_MAX : r; }
u16 TARGET_V1 sat_add_u16(u16 a, u16 b) { u16 r; return __builtin_add_overflow(a, b, &r) ? UINT16_MAX : r; }
u16 TARGET_V1 sat_sub_u16(u16 a, u16 b) { u16 r; return __builtin_sub_overflow(a, b, &r) ? 0 : r; }
i16 TARGET_V1 sat_add_i16(i16 a, i16 b) { i16 r; return __builtin_add_overflow(a, b, &r) ? a < 0 ? INT16_MIN : INT16_MAX : r; }
i16 TARGET_V1 sat_sub_i16(i16 a, i16 b) { i16 r; return __builtin_sub_overflow(a, b, &r) ? a < 0 ? INT16_MIN : INT16_MAX : r; }
u32 TARGET_V1 sat_add_u32(u32 a, u32 b) { u32 r; return __builtin_add_overflow(a, b, &r) ? UINT32_MAX : r; }
u32 TARGET_V1 sat_sub_u32(u32 a, u32 b) { u32 r; return __builtin_sub_overflow(a, b, &r) ? 0 : r; }
i32 TARGET_V1 sat_add_i32(i32 a, i32 b) { i32 r; return __builtin_add_overflow(a, b, &r) ? a < 0 ? INT32_MIN : INT32_MAX : r; }
i32 TARGET_V1 sat_sub_i32(i32 a, i32 b) { i32 r; return __builtin_sub_overflow(a, b, &r) ? a < 0 ? INT32_MIN : INT32_MAX : r; }
u64 TARGET_V1 sat_add_u64(u64 a, u64 b) { u64 r; return __builtin_add_overflow(a, b, &r) ? UINT64_MAX : r; }
u64 TARGET_V1 sat_sub_u64(u64 a, u64 b) { u64 r; return __builtin_sub_overflow(a, b, &r) ? 0 : r; }
i64 TARGET_V1 sat_add_i64(i64 a, i64 b) { i64 r; return __builtin_add_overflow(a, b, &r) ? a < 0 ? INT64_MIN : INT64_MAX : r; }
i64 TARGET_V1 sat_sub_i64(i64 a, i64 b) { i64 r; return __builtin_sub_overflow(a, b, &r) ? a < 0 ? INT64_MIN : INT64_MAX : r; }

// --------------------------
// vector casts
// --------------------------

v8i8 TARGET_V1 trunc_v8i16_8(v8i16 v) { return __builtin_convertvector(v, v8i8); }
v4i16 TARGET_V1 trunc_v4i32_16(v4i32 v) { return __builtin_convertvector(v, v4i16); }
v2i32 TARGET_V1 trunc_v2i64_32(v2i64 v) { return __builtin_convertvector(v, v2i32); }

u8 TARGET_V1 trunc_v8i8_1(v8i8 a) { return (union { v8i1 v; u8 r; }) {.v = __builtin_convertvector((a & 1) != 0, v8i1)}.r; }
u8 TARGET_V1 trunc_v4i16_1(v4i16 a) { return (union { v4i1 v; u8 r; }) {.v = __builtin_convertvector((a & 1) != 0, v4i1)}.r; }
u8 TARGET_V1 trunc_v2i32_1(v2i32 a) { return (union { v2i1 v; u8 r; }) {.v = __builtin_convertvector((a & 1) != 0, v2i1)}.r; }
u16 TARGET_V1 trunc_v16i8_1(v16i8 a) { return (union { v16i1 v; u16 r; }) {.v = __builtin_convertvector((a & 1) != 0, v16i1)}.r; }
u8 TARGET_V1 trunc_v8i16_1(v8i16 a) { return (union { v8i1 v; u8 r; }) {.v = __builtin_convertvector((a & 1) != 0, v8i1)}.r; }
u8 TARGET_V1 trunc_v4i32_1(v4i32 a) { return (union { v4i1 v; u8 r; }) {.v = __builtin_convertvector((a & 1) != 0, v4i1)}.r; }
u8 TARGET_V1 trunc_v2i64_1(v2i64 a) { return (union { v2i1 v; u8 r; }) {.v = __builtin_convertvector((a & 1) != 0, v2i1)}.r; }

// --------------------------
// vector integer arithmetic
// --------------------------

v8u8 TARGET_V1 addv8u8(v8u8 a, v8u8 b) { return (a + b); }
v8u8 TARGET_V1 subv8u8(v8u8 a, v8u8 b) { return (a - b); }
v8u8 TARGET_V1 mulv8u8(v8u8 a, v8u8 b) { return (a * b); }
v8u8 TARGET_V1 landv8u8(v8u8 a, v8u8 b) { return (a & b); }
v8u8 TARGET_V1 lxorv8u8(v8u8 a, v8u8 b) { return (a ^ b); }
v8u8 TARGET_V1 lorv8u8(v8u8 a, v8u8 b) { return (a | b); }
v8u8 TARGET_V1 shlv8u8(v8u8 a, v8u8 b) { return (a << b); }
v8u8 TARGET_V1 shrv8u8(v8u8 a, v8u8 b) { return (a >> b); }
v8i8 TARGET_V1 ashrv8i8(v8i8 a, v8i8 b) { return (a >> b); }
v4u16 TARGET_V1 addv4u16(v4u16 a, v4u16 b) { return (a + b); }
v4u16 TARGET_V1 subv4u16(v4u16 a, v4u16 b) { return (a - b); }
v4u16 TARGET_V1 mulv4u16(v4u16 a, v4u16 b) { return (a * b); }
v4u16 TARGET_V1 landv4u16(v4u16 a, v4u16 b) { return (a & b); }
v4u16 TARGET_V1 lxorv4u16(v4u16 a, v4u16 b) { return (a ^ b); }
v4u16 TARGET_V1 lorv4u16(v4u16 a, v4u16 b) { return (a | b); }
v4u16 TARGET_V1 shlv4u16(v4u16 a, v4u16 b) { return (a << b); }
v4u16 TARGET_V1 shrv4u16(v4u16 a, v4u16 b) { return (a >> b); }
v4i16 TARGET_V1 ashrv4i16(v4i16 a, v4i16 b) { return (a >> b); }
v2u32 TARGET_V1 addv2u32(v2u32 a, v2u32 b) { return (a + b); }
v2u32 TARGET_V1 subv2u32(v2u32 a, v2u32 b) { return (a - b); }
v2u32 TARGET_V1 mulv2u32(v2u32 a, v2u32 b) { return (a * b); }
v2u32 TARGET_V1 landv2u32(v2u32 a, v2u32 b) { return (a & b); }
v2u32 TARGET_V1 lxorv2u32(v2u32 a, v2u32 b) { return (a ^ b); }
v2u32 TARGET_V1 lorv2u32(v2u32 a, v2u32 b) { return (a | b); }
v2u32 TARGET_V1 shlv2u32(v2u32 a, v2u32 b) { return (a << b); }
v2u32 TARGET_V1 shrv2u32(v2u32 a, v2u32 b) { return (a >> b); }
v2i32 TARGET_V1 ashrv2i32(v2i32 a, v2i32 b) { return (a >> b); }
v16u8 TARGET_V1 addv16u8(v16u8 a, v16u8 b) { return (a + b); }
v16u8 TARGET_V1 subv16u8(v16u8 a, v16u8 b) { return (a - b); }
v16u8 TARGET_V1 mulv16u8(v16u8 a, v16u8 b) { return (a * b); }
v16u8 TARGET_V1 landv16u8(v16u8 a, v16u8 b) { return (a & b); }
v16u8 TARGET_V1 lxorv16u8(v16u8 a, v16u8 b) { return (a ^ b); }
v16u8 TARGET_V1 lorv16u8(v16u8 a, v16u8 b) { return (a | b); }
v16u8 TARGET_V1 shlv16u8(v16u8 a, v16u8 b) { return (a << b); }
v16u8 TARGET_V1 shrv16u8(v16u8 a, v16u8 b) { return (a >> b); }
v16i8 TARGET_V1 ashrv16i8(v16i8 a, v16i8 b) { return (a >> b); }
v8u16 TARGET_V1 addv8u16(v8u16 a, v8u16 b) { return (a + b); }
v8u16 TARGET_V1 subv8u16(v8u16 a, v8u16 b) { return (a - b); }
v8u16 TARGET_V1 mulv8u16(v8u16 a, v8u16 b) { return (a * b); }
v8u16 TARGET_V1 landv8u16(v8u16 a, v8u16 b) { return (a & b); }
v8u16 TARGET_V1 lxorv8u16(v8u16 a, v8u16 b) { return (a ^ b); }
v8u16 TARGET_V1 lorv8u16(v8u16 a, v8u16 b) { return (a | b); }
v8u16 TARGET_V1 shlv8u16(v8u16 a, v8u16 b) { return (a << b); }
v8u16 TARGET_V1 shrv8u16(v8u16 a, v8u16 b) { return (a >> b); }
v8i16 TARGET_V1 ashrv8i16(v8i16 a, v8i16 b) { return (a >> b); }
v4u32 TARGET_V1 addv4u32(v4u32 a, v4u32 b) { return (a + b); }
v4u32 TARGET_V1 subv4u32(v4u32 a, v4u32 b) { return (a - b); }
v4u32 TARGET_V1 mulv4u32(v4u32 a, v4u32 b) { return (a * b); }
v4u32 TARGET_V1 landv4u32(v4u32 a, v4u32 b) { return (a & b); }
v4u32 TARGET_V1 lxorv4u32(v4u32 a, v4u32 b) { return (a ^ b); }
v4u32 TARGET_V1 lorv4u32(v4u32 a, v4u32 b) { return (a | b); }
v4u32 TARGET_V1 shlv4u32(v4u32 a, v4u32 b) { return (a << b); }
v4u32 TARGET_V1 shrv4u32(v4u32 a, v4u32 b) { return (a >> b); }
v4i32 TARGET_V1 ashrv4i32(v4i32 a, v4i32 b) { return (a >> b); }
v2u64 TARGET_V1 addv2u64(v2u64 a, v2u64 b) { return (a + b); }
v2u64 TARGET_V1 subv2u64(v2u64 a, v2u64 b) { return (a - b); }
v2u64 TARGET_V1 mulv2u64(v2u64 a, v2u64 b) { return (a * b); }
v2u64 TARGET_V1 landv2u64(v2u64 a, v2u64 b) { return (a & b); }
v2u64 TARGET_V1 lxorv2u64(v2u64 a, v2u64 b) { return (a ^ b); }
v2u64 TARGET_V1 lorv2u64(v2u64 a, v2u64 b) { return (a | b); }
v2u64 TARGET_V1 shlv2u64(v2u64 a, v2u64 b) { return (a << b); }
v2u64 TARGET_V1 shrv2u64(v2u64 a, v2u64 b) { return (a >> b); }
v2i64 TARGET_V1 ashrv2i64(v2i64 a, v2i64 b) { return (a >> b); }

#define ICMP_VEC(pred, cmp, sign, resty, nelem, bits)                          \
    resty TARGET_V1 icmp_##pred##v##nelem##sign##bits(v##nelem##sign##bits a, v##nelem##sign##bits b) { \
      return trunc_##v##nelem##i##bits##_1(a cmp b);                           \
    }                                                                          \
    v##nelem##sign##bits TARGET_V1 icmpmask_##pred##v##nelem##sign##bits(v##nelem##sign##bits a, v##nelem##sign##bits b) { \
      return a cmp b;                                                          \
    }                                                                          \
    v##nelem##sign##bits TARGET_V1 icmpset_##pred##v##nelem##sign##bits(v##nelem##sign##bits a, v##nelem##sign##bits b) { \
      return -(a cmp b);                                                       \
    }
#define ICMP_ALL(fn, ...) \
    fn(eq, ==, u, __VA_ARGS__) \
    fn(ne, !=, u, __VA_ARGS__) \
    fn(ugt, >, u, __VA_ARGS__) \
    fn(uge, >=, u, __VA_ARGS__) \
    fn(ult, <, u, __VA_ARGS__) \
    fn(ule, <=, u, __VA_ARGS__) \
    fn(sgt, >, i, __VA_ARGS__) \
    fn(sge, >=, i, __VA_ARGS__) \
    fn(slt, <, i, __VA_ARGS__) \
    fn(sle, <=, i, __VA_ARGS__)

ICMP_ALL(ICMP_VEC, u8, 8, 8)
ICMP_ALL(ICMP_VEC, u8, 4, 16)
ICMP_ALL(ICMP_VEC, u8, 2, 32)
ICMP_ALL(ICMP_VEC, u16, 16, 8)
ICMP_ALL(ICMP_VEC, u8, 8, 16)
ICMP_ALL(ICMP_VEC, u8, 4, 32)
ICMP_ALL(ICMP_VEC, u8, 2, 64)

u64 TARGET_V1 insert_vi1(u64 v, unsigned n, bool e) { return v & ~((u64)1 << n) | ((u64)e << n); }

// --------------------------
// float arithmetic
// --------------------------

float TARGET_V1 addf32(float a, float b) { return (a + b); }
float TARGET_V1 subf32(float a, float b) { return (a - b); }
float TARGET_V1 mulf32(float a, float b) { return (a * b); }
float TARGET_V1 divf32(float a, float b) { return (a / b); }
//float TARGET_V1 remf32(float a, float b) { return __builtin_fmodf(a, b); }

double TARGET_V1 addf64(double a, double b) { return (a + b); }
double TARGET_V1 subf64(double a, double b) { return (a - b); }
double TARGET_V1 mulf64(double a, double b) { return (a * b); }
double TARGET_V1 divf64(double a, double b) { return (a / b); }
//double TARGET_V1 remf64(double a, double b) { return __builtin_fmod(a, b); }

v2f32 TARGET_V1 addv2f32(v2f32 a, v2f32 b) { return (a + b); }
v2f32 TARGET_V1 subv2f32(v2f32 a, v2f32 b) { return (a - b); }
v2f32 TARGET_V1 mulv2f32(v2f32 a, v2f32 b) { return (a * b); }
v2f32 TARGET_V1 divv2f32(v2f32 a, v2f32 b) { return (a / b); }

v4f32 TARGET_V1 addv4f32(v4f32 a, v4f32 b) { return (a + b); }
v4f32 TARGET_V1 subv4f32(v4f32 a, v4f32 b) { return (a - b); }
v4f32 TARGET_V1 mulv4f32(v4f32 a, v4f32 b) { return (a * b); }
v4f32 TARGET_V1 divv4f32(v4f32 a, v4f32 b) { return (a / b); }

v2f64 TARGET_V1 addv2f64(v2f64 a, v2f64 b) { return (a + b); }
v2f64 TARGET_V1 subv2f64(v2f64 a, v2f64 b) { return (a - b); }
v2f64 TARGET_V1 mulv2f64(v2f64 a, v2f64 b) { return (a * b); }
v2f64 TARGET_V1 divv2f64(v2f64 a, v2f64 b) { return (a / b); }

float TARGET_V1 fnegf32(float a) { return (-a); }
double TARGET_V1 fnegf64(double a) { return (-a); }
fp128 TARGET_V1 fnegf128(fp128 a) { return -a; }
v2f32 TARGET_V1 fnegv2f32(v2f32 a) { return (-a); }
v4f32 TARGET_V1 fnegv4f32(v4f32 a) { return (-a); }
v2f64 TARGET_V1 fnegv2f64(v2f64 a) { return (-a); }

float TARGET_V1 fabsf32(float a) { return __builtin_fabsf(a); }
double TARGET_V1 fabsf64(double a) { return __builtin_fabs(a); }
fp128 TARGET_V1 fabsf128(fp128 a) { return __builtin_fabsf128(a); }

float TARGET_V1 fmaf32(float a, float b, float c) { return a * b + c; }
double TARGET_V1 fmaf64(double a, double b, double c) { return a * b + c; }

float TARGET_V1 copysignf32(float a, float b) { return __builtin_copysignf(a, b); }
double TARGET_V1 copysignf64(double a, double b) { return __builtin_copysign(a, b); }

float TARGET_V1 sqrtf32(float a) { return __builtin_sqrtf(a); }
double TARGET_V1 sqrtf64(double a) { return __builtin_sqrt(a); }

float TARGET_V1 minnumf32(float a, float b) { return __builtin_fminf(a, b); }
double TARGET_V1 minnumf64(double a, double b) { return __builtin_fmin(a, b); }
float TARGET_V1 maxnumf32(float a, float b) { return __builtin_fmaxf(a, b); }
double TARGET_V1 maxnumf64(double a, double b) { return __builtin_fmax(a, b); }
// TODO: enable when Clang exposes llvm.minimum/maximum
//float TARGET_V1 minimumf32(float a, float b) { return __builtin_fminimumf(a, b); }
//double TARGET_V1 minimumf64(double a, double b) { return __builtin_fminimum(a, b); }
//float TARGET_V1 maximumf32(float a, float b) { return __builtin_fmaximumf(a, b); }
//double TARGET_V1 maximumf64(double a, double b) { return __builtin_fmaximum(a, b); }
// TODO: enable in C23 mode when Clang exposes llvm.minimumnum/maximumnum
//float TARGET_V1 minimumnumf32(float a, float b) { return __builtin_fminimum_numf(a, b); }
//double TARGET_V1 minimumnumf64(double a, double b) { return __builtin_fminimum_num(a, b); }
//float TARGET_V1 maximumnumf32(float a, float b) { return __builtin_fmaximum_numf(a, b); }
//double TARGET_V1 maximumnumf64(double a, double b) { return __builtin_fmaximum_num(a, b); }

// --------------------------
// float conversions
// --------------------------

float TARGET_V1 f64tof32(double a) { return (float)(a); }
double TARGET_V1 f32tof64(float a) { return (double)(a); }

i32 TARGET_V1 f32toi32(float a) { return (i32)a; }
u32 TARGET_V1 f32tou32(float a) { return (u32)a; }
i64 TARGET_V1 f32toi64(float a) { return (i64)a; }
u64 TARGET_V1 f32tou64(float a) { return (u64)a; }
i32 TARGET_V1 f64toi32(double a) { return (i32)a; }
u32 TARGET_V1 f64tou32(double a) { return (u32)a; }
i64 TARGET_V1 f64toi64(double a) { return (i64)a; }
u64 TARGET_V1 f64tou64(double a) { return (u64)a; }

// Clang exposes these only under -fno-strict-float-cast-overflow, which would
// inhibit better code generation for non-saturating conversions on x86-64.
i32 TARGET_V1 llvm_fptosi_sat_i32_float(float) __asm__("llvm.fptosi.sat.i32.f32");
i32 TARGET_V1 f32toi32_sat(float a) { return llvm_fptosi_sat_i32_float(a); }
u32 TARGET_V1 llvm_fptoui_sat_i32_float(float) __asm__("llvm.fptoui.sat.i32.f32");
u32 TARGET_V1 f32tou32_sat(float a) { return llvm_fptoui_sat_i32_float(a); }
i64 TARGET_V1 llvm_fptosi_sat_i64_float(float) __asm__("llvm.fptosi.sat.i64.f32");
i64 TARGET_V1 f32toi64_sat(float a) { return llvm_fptosi_sat_i64_float(a); }
u64 TARGET_V1 llvm_fptoui_sat_i64_float(float) __asm__("llvm.fptoui.sat.i64.f32");
u64 TARGET_V1 f32tou64_sat(float a) { return llvm_fptoui_sat_i64_float(a); }
i32 TARGET_V1 llvm_fptosi_sat_i32_double(double) __asm__("llvm.fptosi.sat.i32.f64");
i32 TARGET_V1 f64toi32_sat(double a) { return llvm_fptosi_sat_i32_double(a); }
u32 TARGET_V1 llvm_fptoui_sat_i32_double(double) __asm__("llvm.fptoui.sat.i32.f64");
u32 TARGET_V1 f64tou32_sat(double a) { return llvm_fptoui_sat_i32_double(a); }
i64 TARGET_V1 llvm_fptosi_sat_i64_double(double) __asm__("llvm.fptosi.sat.i64.f64");
i64 TARGET_V1 f64toi64_sat(double a) { return llvm_fptosi_sat_i64_double(a); }
u64 TARGET_V1 llvm_fptoui_sat_i64_double(double) __asm__("llvm.fptoui.sat.i64.f64");
u64 TARGET_V1 f64tou64_sat(double a) { return llvm_fptoui_sat_i64_double(a); }

float TARGET_V1 i8tof32(u8 a) { return (float)(i8)a; }
float TARGET_V1 i16tof32(u16 a) { return (float)(i16)a; }
float TARGET_V1 i32tof32(u32 a) { return (float)(i32)a; }
float TARGET_V1 i64tof32(u64 a) { return (float)(i64)a; }
float TARGET_V1 u8tof32(i8 a) { return (float)(u8)a; }
float TARGET_V1 u16tof32(i16 a) { return (float)(u16)a; }
float TARGET_V1 u32tof32(i32 a) { return (float)(u32)a; }
float TARGET_V1 u64tof32(i64 a) { return (float)(u64)a; }

double TARGET_V1 i8tof64(u8 a) { return (double)(i8)a; }
double TARGET_V1 i16tof64(u16 a) { return (double)(i16)a; }
double TARGET_V1 i32tof64(u32 a) { return (double)(i32)a; }
double TARGET_V1 i64tof64(u64 a) { return (double)(i64)a; }
double TARGET_V1 u8tof64(i8 a) { return (double)(u8)a; }
double TARGET_V1 u16tof64(i16 a) { return (double)(u16)a; }
double TARGET_V1 u32tof64(i32 a) { return (double)(u32)a; }
double TARGET_V1 u64tof64(i64 a) { return (double)(u64)a; }

// --------------------------
// extensions
// --------------------------

i64 TARGET_V1 fill_with_sign64(i64 a) { return (a >> 63); }

// --------------------------
// atomics
// --------------------------

typedef struct CmpXchgRes { u64 orig; bool success; } CmpXchgRes;

#define CMPXCHG(ty) \
  CmpXchgRes TARGET_V1 cmpxchg_##ty##_monotonic(ty* ptr, ty cmp, ty new_val) { \
      bool res = __atomic_compare_exchange_n(ptr, &cmp, new_val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED); \
      return (CmpXchgRes){cmp, res}; \
  } \
  CmpXchgRes TARGET_V1 cmpxchg_##ty##_acquire(ty* ptr, ty cmp, ty new_val) { \
      bool res = __atomic_compare_exchange_n(ptr, &cmp, new_val, false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE); \
      return (CmpXchgRes){cmp, res}; \
  } \
  CmpXchgRes TARGET_V1 cmpxchg_##ty##_release(ty* ptr, ty cmp, ty new_val) { \
      bool res = __atomic_compare_exchange_n(ptr, &cmp, new_val, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED); \
      return (CmpXchgRes){cmp, res}; \
  } \
  CmpXchgRes TARGET_V1 cmpxchg_##ty##_acqrel(ty* ptr, ty cmp, ty new_val) { \
      bool res = __atomic_compare_exchange_n(ptr, &cmp, new_val, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE); \
      return (CmpXchgRes){cmp, res}; \
  } \
  CmpXchgRes TARGET_V1 cmpxchg_##ty##_seqcst(ty* ptr, ty cmp, ty new_val) { \
      bool res = __atomic_compare_exchange_n(ptr, &cmp, new_val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); \
      return (CmpXchgRes){cmp, res}; \
  }

CMPXCHG(u8)
CMPXCHG(u16)
CMPXCHG(u32)
CMPXCHG(u64)

#undef CMPXCHG

u32 TARGET_V1 atomic_load_u8_mono(u8* ptr) { return __atomic_load_n(ptr, __ATOMIC_RELAXED); }
u32 TARGET_V1 atomic_load_u16_mono(u16* ptr) { return __atomic_load_n(ptr, __ATOMIC_RELAXED); }
u32 TARGET_V1 atomic_load_u32_mono(u32* ptr) { return __atomic_load_n(ptr, __ATOMIC_RELAXED); }
u64 TARGET_V1 atomic_load_u64_mono(u64* ptr) { return __atomic_load_n(ptr, __ATOMIC_RELAXED); }
u32 TARGET_V1 atomic_load_u8_acq(u8* ptr) { return __atomic_load_n(ptr, __ATOMIC_ACQUIRE); }
u32 TARGET_V1 atomic_load_u16_acq(u16* ptr) { return __atomic_load_n(ptr, __ATOMIC_ACQUIRE); }
u32 TARGET_V1 atomic_load_u32_acq(u32* ptr) { return __atomic_load_n(ptr, __ATOMIC_ACQUIRE); }
u64 TARGET_V1 atomic_load_u64_acq(u64* ptr) { return __atomic_load_n(ptr, __ATOMIC_ACQUIRE); }
u32 TARGET_V1 atomic_load_u8_seqcst(u8* ptr) { return __atomic_load_n(ptr, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_load_u16_seqcst(u16* ptr) { return __atomic_load_n(ptr, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_load_u32_seqcst(u32* ptr) { return __atomic_load_n(ptr, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_load_u64_seqcst(u64* ptr) { return __atomic_load_n(ptr, __ATOMIC_SEQ_CST); }

void TARGET_V1 atomic_store_u8_mono(u8* ptr, u8 v) { __atomic_store_n(ptr, v, __ATOMIC_RELAXED); }
void TARGET_V1 atomic_store_u16_mono(u16* ptr, u16 v) { __atomic_store_n(ptr, v, __ATOMIC_RELAXED); }
void TARGET_V1 atomic_store_u32_mono(u32* ptr, u32 v) { __atomic_store_n(ptr, v, __ATOMIC_RELAXED); }
void TARGET_V1 atomic_store_u64_mono(u64* ptr, u64 v) { __atomic_store_n(ptr, v, __ATOMIC_RELAXED); }
void TARGET_V1 atomic_store_u8_rel(u8* ptr, u8 v) { __atomic_store_n(ptr, v, __ATOMIC_RELEASE); }
void TARGET_V1 atomic_store_u16_rel(u16* ptr, u16 v) { __atomic_store_n(ptr, v, __ATOMIC_RELEASE); }
void TARGET_V1 atomic_store_u32_rel(u32* ptr, u32 v) { __atomic_store_n(ptr, v, __ATOMIC_RELEASE); }
void TARGET_V1 atomic_store_u64_rel(u64* ptr, u64 v) { __atomic_store_n(ptr, v, __ATOMIC_RELEASE); }
void TARGET_V1 atomic_store_u8_seqcst(u8* ptr, u8 v) { __atomic_store_n(ptr, v, __ATOMIC_SEQ_CST); }
void TARGET_V1 atomic_store_u16_seqcst(u16* ptr, u16 v) { __atomic_store_n(ptr, v, __ATOMIC_SEQ_CST); }
void TARGET_V1 atomic_store_u32_seqcst(u32* ptr, u32 v) { __atomic_store_n(ptr, v, __ATOMIC_SEQ_CST); }
void TARGET_V1 atomic_store_u64_seqcst(u64* ptr, u64 v) { __atomic_store_n(ptr, v, __ATOMIC_SEQ_CST); }

u32 TARGET_V1 atomic_xchg_u8_seqcst(u8 *p, u8 v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_add_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_sub_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_and_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_nand_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_nand(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_or_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_xor_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_xor(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_min_i8_seqcst(i8 *p, i8 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_max_i8_seqcst(i8 *p, i8 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_min_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_max_u8_seqcst(u8 *p, u8 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_xchg_u16_seqcst(u16 *p, u16 v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_add_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_sub_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_and_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_nand_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_nand(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_or_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_xor_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_xor(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_min_i16_seqcst(i16 *p, i16 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_max_i16_seqcst(i16 *p, i16 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_min_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_max_u16_seqcst(u16 *p, u16 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_xchg_u32_seqcst(u32 *p, u32 v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_add_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_sub_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_and_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_nand_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_nand(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_or_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_xor_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_xor(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_min_i32_seqcst(i32 *p, i32 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_max_i32_seqcst(i32 *p, i32 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_min_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u32 TARGET_V1 atomic_max_u32_seqcst(u32 *p, u32 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_xchg_u64_seqcst(u64 *p, u64 v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_add_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_sub_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_and_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_nand_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_nand(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_or_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_xor_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_xor(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_min_i64_seqcst(i64 *p, i64 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_max_i64_seqcst(i64 *p, i64 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_min_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
u64 TARGET_V1 atomic_max_u64_seqcst(u64 *p, u64 v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
float TARGET_V1 atomic_add_f32_seqcst(float *p, float v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
float TARGET_V1 atomic_sub_f32_seqcst(float *p, float v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
float TARGET_V1 atomic_min_f32_seqcst(float *p, float v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
float TARGET_V1 atomic_max_f32_seqcst(float *p, float v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
double TARGET_V1 atomic_add_f64_seqcst(double *p, double v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
double TARGET_V1 atomic_sub_f64_seqcst(double *p, double v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
double TARGET_V1 atomic_min_f64_seqcst(double *p, double v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
double TARGET_V1 atomic_max_f64_seqcst(double *p, double v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }

void TARGET_V1 fence_acq(void) { __atomic_thread_fence(__ATOMIC_ACQUIRE); }
void TARGET_V1 fence_rel(void) { __atomic_thread_fence(__ATOMIC_RELEASE); }
void TARGET_V1 fence_acqrel(void) { __atomic_thread_fence(__ATOMIC_ACQ_REL); }
void TARGET_V1 fence_seqcst(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

// --------------------------
// select
// --------------------------

i32 TARGET_V1 select_i32(u8 cond, i32 val1, i32 val2) { return ((cond & 1) ? val1 : val2); }
i64 TARGET_V1 select_i64(u8 cond, i64 val1, i64 val2) { return ((cond & 1) ? val1 : val2); }
i128 TARGET_V1 select_i128(u8 cond, i128 val1, i128 val2) { return ((cond & 1) ? val1 : val2); }
float TARGET_V1 select_f32(u8 cond, float val1, float val2) { return ((cond & 1) ? val1 : val2); }
double TARGET_V1 select_f64(u8 cond, double val1, double val2) { return ((cond & 1) ? val1 : val2); }
v2u64 TARGET_V1 select_v2u64(u8 cond, v2u64 val1, v2u64 val2) { return ((cond & 1) ? val1 : val2); }

// --------------------------
// float comparisons
// --------------------------

#define FOP_ORD(ty, name, op) u32 TARGET_V1 fcmp_##name##_##ty(ty a, ty b) { return !__builtin_isunordered(a, b) && (a op b); }
#define FOP_UNRD(ty, name, op) u32 TARGET_V1 fcmp_##name##_##ty(ty a, ty b) { return __builtin_isunordered(a, b) || (a op b); }
#define FOPS(ty) FOP_ORD(ty, oeq, ==) \
    FOP_ORD(ty, ogt, >) \
    FOP_ORD(ty, oge, >=) \
    FOP_ORD(ty, olt, <) \
    FOP_ORD(ty, ole, <=) \
    FOP_ORD(ty, one, !=) \
    u32 TARGET_V1 fcmp_ord_##ty(ty a, ty b) { return !__builtin_isunordered(a, b); } \
    FOP_UNRD(ty, ueq, ==) \
    FOP_UNRD(ty, ugt, >) \
    FOP_UNRD(ty, uge, >=) \
    FOP_UNRD(ty, ult, <) \
    FOP_UNRD(ty, ule, <=) \
    FOP_UNRD(ty, une, !=) \
    u32 TARGET_V1 fcmp_uno_##ty(ty a, ty b) { return __builtin_isunordered(a, b); }

FOPS(float)
FOPS(double)


#undef FOP_ORD
#undef FOP_UNORD
#undef FOPS

// --------------------------
// is_fpclass
// --------------------------

#define FOP(ty, name, num) u8 TARGET_V1 is_fpclass_##name##_##ty(u8 c, ty a) { return c | __builtin_isfpclass(a, (num)); }
#define FOPS(ty) \
    FOP(ty, snan, 1<<0) \
    FOP(ty, qnan, 1<<1) \
    FOP(ty, ninf, 1<<2) \
    FOP(ty, nnorm, 1<<3) \
    FOP(ty, nsnorm, 1<<4) \
    FOP(ty, nzero, 1<<5) \
    FOP(ty, pzero, 1<<6) \
    FOP(ty, psnorm, 1<<7) \
    FOP(ty, pnorm, 1<<8) \
    FOP(ty, pinf, 1<<9) \
    FOP(ty, nan, (1<<0)|(1<<1)) \
    FOP(ty, inf, (1<<2)|(1<<9)) \
    FOP(ty, norm, (1<<3)|(1<<8)) \
    FOP(ty, finite, (1<<3)|(1<<4)|(1<<5)|(1<<6)|(1<<7)|(1<<8))

FOPS(float)
FOPS(double)

#undef FOPS
#undef FOP

// --------------------------
// miscellaneous
// --------------------------

void TARGET_V1 trap(void) { __builtin_trap(); }
void TARGET_V1 debugtrap(void) { __builtin_debugtrap(); }

void prefetch_rl0(void* addr) { __builtin_prefetch(addr, 0, 0); }
void prefetch_rl1(void* addr) { __builtin_prefetch(addr, 0, 1); }
void prefetch_rl2(void* addr) { __builtin_prefetch(addr, 0, 2); }
void prefetch_rl3(void* addr) { __builtin_prefetch(addr, 0, 3); }

void prefetch_wl0(void* addr) { __builtin_prefetch(addr, 1, 0); }
void prefetch_wl1(void* addr) { __builtin_prefetch(addr, 1, 1); }
void prefetch_wl2(void* addr) { __builtin_prefetch(addr, 1, 2); }
void prefetch_wl3(void* addr) { __builtin_prefetch(addr, 1, 3); }
