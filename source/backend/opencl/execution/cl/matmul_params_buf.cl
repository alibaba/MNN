#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// =================================================================================================
#define USE_INLINE_KEYWORD 1

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef GEMMK
  #define GEMMK 0    // Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
  #define KWG 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
  #define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
  #define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
  #define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
  #define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
  #define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
  #define VWM 1      // Vector width of matrices A and C
#endif
#ifndef VWN
  #define VWN 1      // Vector width of matrix B
#endif
#ifndef STRM
  #define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
  #define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
  #define SA 0       // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
  #define SB 0       // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
  #define KREG 1     // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
  #define USE_VECTOR_MAD 0      // Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
  #define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance
#endif

// Half-precision
#if PRECISION == 16
  typedef half real;
  typedef half2 real2;
  typedef half4 real4;
  typedef half8 real8;
  typedef half16 real16;
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
  typedef float real_arg;
  #define GetRealArg(x) (half)x
#else
  typedef real real_arg;
  #define GetRealArg(x) x
#endif

// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
  #define LOCAL_PTR __local
#endif

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size
// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
#ifndef RELAX_WORKGROUP_SIZE
  #define RELAX_WORKGROUP_SIZE 0
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a)
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
  #define IsZero(a) (a == ZERO)
#endif

// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)
#else
  #define AbsoluteValue(value) value = fabs(value)
#endif

// Negation (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define Negate(value) value.x = -(value.x); value.y = -(value.y)
#else
  #define Negate(value) value = -(value)
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y
#else
  #define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a,b) a.x*b.x - a.y*b.y
  #define MulImag(a,b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c,a,b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #if USE_CL_MAD == 1
    #define MultiplyAdd(c,a,b) c = mad(a, b, c)
  #else
    #define MultiplyAdd(c,a,b) c += a * b
  #endif
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)
#else
  #define MultiplySubtract(c,a,b) c -= a * b
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
  #define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom
#else
  #define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value)
#endif

// =================================================================================================

// Force inlining functions or not: some compilers don't support the inline keyword
#ifdef USE_INLINE_KEYWORD
  #define INLINE_FUNC inline
#else
  #define INLINE_FUNC
#endif

// =================================================================================================

  INLINE_FUNC int GetGroupID1() { return get_group_id(1); }
  INLINE_FUNC int GetGroupID0() { return get_group_id(0); }

// =================================================================================================

// End of the C++11 raw string literal

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#elif VWM == 16
    typedef real16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
#elif VWN == 2
    typedef real2 realN;
#elif VWN == 4
    typedef real4 realN;
#elif VWN == 8
    typedef real8 realN;
#elif VWN == 16
    typedef real16 realN;
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
INLINE_FUNC realM InitAccRegisters() {
  realM result;
  #if VWM == 1
    SetToZero(result);
  #elif VWM == 2
    SetToZero(result.x);
    SetToZero(result.y);
  #elif VWM == 4
    SetToZero(result.x);
    SetToZero(result.y);
    SetToZero(result.z);
    SetToZero(result.w);
  #elif VWM == 8
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
  #elif VWM == 16
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
    SetToZero(result.s8);
    SetToZero(result.s9);
    SetToZero(result.sA);
    SetToZero(result.sB);
    SetToZero(result.sC);
    SetToZero(result.sD);
    SetToZero(result.sE);
    SetToZero(result.sF);
  #endif
  return result;
}

INLINE_FUNC realN InitAccRegistersN() {
  realN result;
  #if VWN == 1
    SetToZero(result);
  #elif VWN == 2
    SetToZero(result.x);
    SetToZero(result.y);
  #elif VWN == 4
    SetToZero(result.x);
    SetToZero(result.y);
    SetToZero(result.z);
    SetToZero(result.w);
  #elif VWN == 8
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
  #elif VWN == 16
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
    SetToZero(result.s8);
    SetToZero(result.s9);
    SetToZero(result.sA);
    SetToZero(result.sB);
    SetToZero(result.sC);
    SetToZero(result.sD);
    SetToZero(result.sE);
    SetToZero(result.sF);
  #endif
  return result;
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
INLINE_FUNC void GlobalToLocalA(const __global realM* restrict agm, LOCAL_PTR realM* alm,
                                const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = _mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + _mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = _kia + la1*KWA;
      int idm = mg + GetGroupID0() * (MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC void GlobalToLocalB(const __global realN* restrict bgm, LOCAL_PTR realN* blm,
                                const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = _nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + _nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = _kib + lb1*KWB;
      int idn = ng + GetGroupID1() * (NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];
    }
  }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
INLINE_FUNC realM GlobalToPrivateA(const __global realM* restrict agm, const int _mi,
                                   const int kSizeM, const int idk, const int kwg) {
  // Computes the indices based on strided/non-strided access
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif

  // Computes the indices for the global memory
  int idm = mg + GetGroupID0() * (MWG/VWM);

  // Loads the data from global memory (not transposed) and stores into registers
  return agm[idk*(kSizeM/VWM) + idm];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
INLINE_FUNC realN GlobalToPrivateB(const __global realN* restrict bgm, const int _ni,
                                   const int kSizeN, const int idk) {
  // Computes the indices based on strided/non-strided access
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif

  // Computes the indices for the global memory
  int idn = ng + GetGroupID1() * (NWG/VWN);

  // Loads the data from global memory (transposed) and stores into registers
  return bgm[idk*(kSizeN/VWN) + idn];
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
INLINE_FUNC realM LocalToPrivateA(LOCAL_PTR realM* alm, const int _mi, const int kg) {
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif
  return alm[kg*(MWG/VWM) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC realN LocalToPrivateB(LOCAL_PTR realN* blm, const int _ni, const int kg) {
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif
  return blm[kg*(NWG/VWN) + ng];
}
#endif



// The vectorised multiply-add function
INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWM == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWM == 2
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
    #elif VWM == 4
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
      MultiplyAdd(cvec.z , avec.z,  bval);
      MultiplyAdd(cvec.w , avec.w,  bval);
    #elif VWM == 8
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
    #elif VWM == 16
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
      MultiplyAdd(cvec.s8, avec.s8, bval);
      MultiplyAdd(cvec.s9, avec.s9, bval);
      MultiplyAdd(cvec.sA, avec.sA, bval);
      MultiplyAdd(cvec.sB, avec.sB, bval);
      MultiplyAdd(cvec.sC, avec.sC, bval);
      MultiplyAdd(cvec.sD, avec.sD, bval);
      MultiplyAdd(cvec.sE, avec.sE, bval);
      MultiplyAdd(cvec.sF, avec.sF, bval);
    #endif
  #endif
  return cvec;
}

// The vectorised multiply-add function
INLINE_FUNC realN MultiplyAddVectorN(realN cvec, const real avec, const realN bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWN == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWN == 2
      MultiplyAdd(cvec.x , avec,  bval.x);
      MultiplyAdd(cvec.y , avec,  bval.y);
    #elif VWN == 4
      MultiplyAdd(cvec.x , avec,  bval.x);
      MultiplyAdd(cvec.y , avec,  bval.y);
      MultiplyAdd(cvec.z , avec,  bval.z);
      MultiplyAdd(cvec.w , avec,  bval.w);
    #elif VWN == 8
      MultiplyAdd(cvec.s0, avec, bval.s0);
      MultiplyAdd(cvec.s1, avec, bval.s1);
      MultiplyAdd(cvec.s2, avec, bval.s2);
      MultiplyAdd(cvec.s3, avec, bval.s3);
      MultiplyAdd(cvec.s4, avec, bval.s4);
      MultiplyAdd(cvec.s5, avec, bval.s5);
      MultiplyAdd(cvec.s6, avec, bval.s6);
      MultiplyAdd(cvec.s7, avec, bval.s7);
    #elif VWN == 16
      MultiplyAdd(cvec.s0, avec, bval.s0);
      MultiplyAdd(cvec.s1, avec, bval.s1);
      MultiplyAdd(cvec.s2, avec, bval.s2);
      MultiplyAdd(cvec.s3, avec, bval.s3);
      MultiplyAdd(cvec.s4, avec, bval.s4);
      MultiplyAdd(cvec.s5, avec, bval.s5);
      MultiplyAdd(cvec.s6, avec, bval.s6);
      MultiplyAdd(cvec.s7, avec, bval.s7);
      MultiplyAdd(cvec.s8, avec, bval.s8);
      MultiplyAdd(cvec.s9, avec, bval.s9);
      MultiplyAdd(cvec.sA, avec, bval.sA);
      MultiplyAdd(cvec.sB, avec, bval.sB);
      MultiplyAdd(cvec.sC, avec, bval.sC);
      MultiplyAdd(cvec.sD, avec, bval.sD);
      MultiplyAdd(cvec.sE, avec, bval.sE);
      MultiplyAdd(cvec.sF, avec, bval.sF);
    #endif
  #endif
  return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
// layout : [N, M]
INLINE_FUNC void StoreResultsM(__global realM* cgm, realM c_value, const int _mi, const int _ni,
                              const int kSizeM, const real alpha, const real beta) {
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif
  #if STRN == 0
    int ng = _ni + get_local_id(1)*NWI;
  #elif STRN == 1
    int ng = _ni%VWN + get_local_id(1)*VWN + (_ni/VWN)*VWN*NDIMC;
  #endif
  int idm = mg + GetGroupID0() * (MWG/VWM);
  int idn = ng + GetGroupID1() * NWG;
  int index = idn*(kSizeM/VWM) + idm;

  realM result;
  realM xval = c_value;

  // The final multiplication with alpha (in case beta == 0)
  if (IsZero(beta)) {
    #if VWM == 1
      Multiply(result, alpha, xval);
    #elif VWM == 2
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
    #elif VWM == 4
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
      Multiply(result.z, alpha, xval.z);
      Multiply(result.w, alpha, xval.w);
    #elif VWM == 8
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
    #elif VWM == 16
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
      Multiply(result.s8, alpha, xval.s8);
      Multiply(result.s9, alpha, xval.s9);
      Multiply(result.sA, alpha, xval.sA);
      Multiply(result.sB, alpha, xval.sB);
      Multiply(result.sC, alpha, xval.sC);
      Multiply(result.sD, alpha, xval.sD);
      Multiply(result.sE, alpha, xval.sE);
      Multiply(result.sF, alpha, xval.sF);
    #endif
  }

  // The final multiplication with alpha and the addition with beta*C
  else {
    realM yval = cgm[index];
    #if VWM == 1
      AXPBY(result, alpha, xval, beta, yval);
    #elif VWM == 2
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
    #elif VWM == 4
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
      AXPBY(result.z, alpha, xval.z, beta, yval.z);
      AXPBY(result.w, alpha, xval.w, beta, yval.w);
    #elif VWM == 8
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
    #elif VWM == 16
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
      AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);
      AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);
      AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);
      AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);
      AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);
      AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);
      AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);
      AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);
    #endif
  }
  cgm[index] = result;
}


// layout : [M, N]
INLINE_FUNC void StoreResultsN(__global realN* cgn, realN c_value,
                            #ifdef BIAS
                            __global realN* egn,
                            #endif
                            const int _mi, const int _ni,
                            const int kSizeN, const real alpha, const real beta) {
                                  
                                  
  #if STRM == 0
    int mg = _mi + get_local_id(0)*MWI;
  #elif STRM == 1
    int mg = _mi%VWM + get_local_id(0)*VWM + (_mi/VWM)*VWM*MDIMC;
  #endif
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif
  int idm = mg + GetGroupID0() * MWG;
  int idn = ng + GetGroupID1() * (NWG/VWN);
  int index = idm * (kSizeN/VWN) + idn;

  realN result = 0;
  realN xval = c_value;
  
  // The final multiplication with alpha (in case beta == 0)
  if (IsZero(beta)) {
    #if VWN == 1
      Multiply(result, alpha, xval);
    #elif VWN == 2
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
    #elif VWN == 4
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
      Multiply(result.z, alpha, xval.z);
      Multiply(result.w, alpha, xval.w);
    #elif VWN == 8
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
    #elif VWN == 16
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
      Multiply(result.s8, alpha, xval.s8);
      Multiply(result.s9, alpha, xval.s9);
      Multiply(result.sA, alpha, xval.sA);
      Multiply(result.sB, alpha, xval.sB);
      Multiply(result.sC, alpha, xval.sC);
      Multiply(result.sD, alpha, xval.sD);
      Multiply(result.sE, alpha, xval.sE);
      Multiply(result.sF, alpha, xval.sF);
    #endif
  }

  // The final multiplication with alpha and the addition with beta*C
  else {
    realN yval = cgn[index];
    #if VWN == 1
      AXPBY(result, alpha, xval, beta, yval);
    #elif VWN == 2
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
    #elif VWN == 4
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
      AXPBY(result.z, alpha, xval.z, beta, yval.z);
      AXPBY(result.w, alpha, xval.w, beta, yval.w);
    #elif VWN == 8
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
    #elif VWN == 16
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
      AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);
      AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);
      AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);
      AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);
      AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);
      AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);
      AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);
      AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);
    #endif
  }
  
  
#ifdef BIAS
  realN xval = egn[idn];

  #if VWN == 1
    result += xval;
  #elif VWN == 2
    result.x += xval.x;
    result.y += xval.y;
  #elif VWN == 4
    result.x += xval.x;
    result.y += xval.y;
    result.z += xval.z;
    result.w += xval.w;
  #elif VWN == 8
    result.s0 += xval.s0;
    result.s1 += xval.s1;
    result.s2 += xval.s2;
    result.s3 += xval.s3;
    result.s4 += xval.s4;
    result.s5 += xval.s5;
    result.s6 += xval.s6;
    result.s7 += xval.s7;
  #elif VWN == 16
    result.s0 += xval.s0;
    result.s1 += xval.s1;
    result.s2 += xval.s2;
    result.s3 += xval.s3;
    result.s4 += xval.s4;
    result.s5 += xval.s5;
    result.s6 += xval.s6;
    result.s7 += xval.s7;
    result.s8 += xval.s8;
    result.s9 += xval.s9;
    result.sA += xval.sA;
    result.sB += xval.sB;
    result.sC += xval.sC;
    result.sD += xval.sD;
    result.sE += xval.sE;
    result.sF += xval.sF;
  #endif
#endif

  cgn[index] = result;
}


// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.
INLINE_FUNC void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                           const __global realM* restrict agm, const __global realN* restrict bgm,
                           #ifdef BIAS
                           const __global realN* restrict egm,
                           #endif
                           __global realM* cgm, const real alpha, const real beta
                           #if SA == 1 && SB == 1
                             , LOCAL_PTR realM* alm, LOCAL_PTR realN* blm
                           #elif SA == 1
                             , LOCAL_PTR realM* alm
                           #elif SB == 1
                             , LOCAL_PTR realN* blm
                           #endif
                           ) {

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  realM apm[MWI/VWM]; // MWI * 1
  #pragma promote_to_registers
  realN bpm[NWI/VWN]; // 1 * NWI

  #ifdef OUTPUTMN
  #pragma promote_to_registers
  realN cpn[MWI*(NWI/VWN)]; // MWI * NWI
  #else
  #pragma promote_to_registers
  realM cpm[NWI*(MWI/VWM)]; // NWI * MWI
  #endif

  // Combined thread identifier (volatile to disable caching)
  #if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  #endif

  // Initializes the accumulation registers
  #ifdef OUTPUTMN
  #pragma unroll
  for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
    #pragma unroll
    for (int _mi = 0; _mi < MWI; _mi += 1) {
      cpn[_mi * (NWI/VWN) + _ni] = InitAccRegistersN();
    }
  }
  #else
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
    }
  }
  #endif

  // Loops over all workgroup tiles
  for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {

    // Loads data: off-chip --> local (matrix A)
    #if SA == 1
      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
    #endif
    // Loads data: off-chip --> local (matrix B)
    #if SB == 1
      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
    #endif
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
      #pragma unroll
      for (int _pit = 0; _pit < KWI*KREG; _pit += KREG) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + _pit;
        #endif
        #if SA == 1 || SB == 1
          int kg = pwi + _pit;
        #endif

        // Loads matrix A (kernel 0) or matrix B (kernel 1)
        #pragma unroll
        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
          // Loads data: local --> private (matrix A)
          #if GEMMK == 0 && SA == 1
            apm[_mi] = LocalToPrivateA(alm, _mi, kg);
          // Loads data: off-chip --> private (matrix A)
          #elif GEMMK == 0 && SA == 0
            apm[_mi] = GlobalToPrivateA(agm, _mi, kSizeM, idk, kwg);
          #endif
        }

        // Loads matrix B (kernel 0) or matrix A (kernel 1)

        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          // Loads data: local --> private (matrix B)
          #if SB == 1
            bpm[_ni] = LocalToPrivateB(blm, _ni, kg);
          // Loads data: off-chip --> private (matrix B)
          #else
            bpm[_ni] = GlobalToPrivateB(bgm, _ni, kSizeN, idk);
          #endif
        }

        // Performs the accumulation (Cpm += Apm * Bpm)

        #ifdef OUTPUTMN
            #pragma unroll
            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
              #pragma unroll
              for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
                const realM aval = apm[_mi];
                #if VWM == 1
                  // [MWI/VWM, VWM, NWI/VWN, VWN]
                  cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni], aval, bpm[_ni]);
                #elif VWM == 2
                  cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni], aval.x, bpm[_ni]);
                  cpn[(_mi*VWM + 1)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 1)*(NWI/VWN) + _ni], aval.y, bpm[_ni]);
                #elif VWM == 4
                  cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni], aval.x, bpm[_ni]);
                  cpn[(_mi*VWM + 1)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 1)*(NWI/VWN) + _ni], aval.y, bpm[_ni]);
                  cpn[(_mi*VWM + 2)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 2)*(NWI/VWN) + _ni], aval.z, bpm[_ni]);
                  cpn[(_mi*VWM + 3)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 3)*(NWI/VWN) + _ni], aval.w, bpm[_ni]);
                #elif VWM == 8
                  cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 0)*(NWI/VWN) + _ni], aval.s0, bpm[_ni]);
                  cpn[(_mi*VWM + 1)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 1)*(NWI/VWN) + _ni], aval.s1, bpm[_ni]);
                  cpn[(_mi*VWM + 2)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 2)*(NWI/VWN) + _ni], aval.s2, bpm[_ni]);
                  cpn[(_mi*VWM + 3)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 3)*(NWI/VWN) + _ni], aval.s3, bpm[_ni]);
                  cpn[(_mi*VWM + 4)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 4)*(NWI/VWN) + _ni], aval.s4, bpm[_ni]);
                  cpn[(_mi*VWM + 5)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 5)*(NWI/VWN) + _ni], aval.s5, bpm[_ni]);
                  cpn[(_mi*VWM + 6)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 6)*(NWI/VWN) + _ni], aval.s6, bpm[_ni]);
                  cpn[(_mi*VWM + 7)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 7)*(NWI/VWN) + _ni], aval.s7, bpm[_ni]);
                #elif VWM == 16
                  cpn[(_mi*VWM + 0 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 0 )*(NWI/VWN) + _ni], aval.s0, bpm[_ni]);
                  cpn[(_mi*VWM + 1 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 1 )*(NWI/VWN) + _ni], aval.s1, bpm[_ni]);
                  cpn[(_mi*VWM + 2 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 2 )*(NWI/VWN) + _ni], aval.s2, bpm[_ni]);
                  cpn[(_mi*VWM + 3 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 3 )*(NWI/VWN) + _ni], aval.s3, bpm[_ni]);
                  cpn[(_mi*VWM + 4 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 4 )*(NWI/VWN) + _ni], aval.s4, bpm[_ni]);
                  cpn[(_mi*VWM + 5 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 5 )*(NWI/VWN) + _ni], aval.s5, bpm[_ni]);
                  cpn[(_mi*VWM + 6 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 6 )*(NWI/VWN) + _ni], aval.s6, bpm[_ni]);
                  cpn[(_mi*VWM + 7 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 7 )*(NWI/VWN) + _ni], aval.s7, bpm[_ni]);
                  cpn[(_mi*VWM + 8 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 8 )*(NWI/VWN) + _ni], aval.s8, bpm[_ni]);
                  cpn[(_mi*VWM + 9 )*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 9 )*(NWI/VWN) + _ni], aval.s9, bpm[_ni]);
                  cpn[(_mi*VWM + 10)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 10)*(NWI/VWN) + _ni], aval.sA, bpm[_ni]);
                  cpn[(_mi*VWM + 11)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 11)*(NWI/VWN) + _ni], aval.sB, bpm[_ni]);
                  cpn[(_mi*VWM + 12)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 12)*(NWI/VWN) + _ni], aval.sC, bpm[_ni]);
                  cpn[(_mi*VWM + 13)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 13)*(NWI/VWN) + _ni], aval.sD, bpm[_ni]);
                  cpn[(_mi*VWM + 14)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 14)*(NWI/VWN) + _ni], aval.sE, bpm[_ni]);
                  cpn[(_mi*VWM + 15)*(NWI/VWN) + _ni] = MultiplyAddVectorN(cpn[(_mi*VWM + 15)*(NWI/VWN) + _ni], aval.sF, bpm[_ni]);
                #endif
              }
            }
        #else
            #pragma unroll
            for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
              #pragma unroll
              for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
                const realM aval = apm[_mi];
                #if VWN == 1
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni]);
                #elif VWN == 2
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
                  cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
                #elif VWN == 4
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
                  cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
                  cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].z);
                  cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].w);
                #elif VWN == 8
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
                  cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
                  cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
                  cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
                  cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
                  cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
                  cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
                  cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
                #elif VWN == 16
                  cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
                  cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
                  cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
                  cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
                  cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
                  cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
                  cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
                  cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
                  cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi], aval, bpm[_ni].s8);
                  cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi], aval, bpm[_ni].s9);
                  cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi], aval, bpm[_ni].sA);
                  cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi], aval, bpm[_ni].sB);
                  cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi], aval, bpm[_ni].sC);
                  cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi], aval, bpm[_ni].sD);
                  cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi], aval, bpm[_ni].sE);
                  cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi], aval, bpm[_ni].sF);
                #endif
              }
            }
        #endif
      }
    }
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif
  }
  #if GLOBAL_MEM_FENCE == 1
    barrier(CLK_GLOBAL_MEM_FENCE);
  #endif

  #ifdef OUTPUTMN
      #pragma unroll
      for (int _mi = 0; _mi < MWI; _mi += 1) {
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          StoreResultsN((__global realN* )cgm, cpn[_mi * (NWI/VWN) + _ni],
              #ifdef BIAS
              egm,
              #endif
              _mi, _ni, kSizeN, alpha, beta);
        }
      }
  
  #else
  
      // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
      const int cld = kSizeM;
      
      #pragma unroll
      for (int _ni = 0; _ni < NWI; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
          StoreResultsM(cgm, cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, alpha, beta);
        }
      }
  #endif
}

// Main entry point of the kernel. This is the regular full version.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void Xgemm(const int kSizeM, const int kSizeN, const int kSizeK,
           const real_arg arg_alpha,
           const real_arg arg_beta,
           const __global realM* restrict agm, // [K, M]
           const __global realN* restrict bgm, // [K, N]
           #ifdef BIAS
           const __global realN* restrict egm, // [N]
           #endif
           __global realM* cgm,
           const int a_offset, const int b_offset, const int c_offset
) {
    const real alpha = GetRealArg(arg_alpha);
    const real beta = GetRealArg(arg_beta);
  
    // Adds the offsets (in case of use of a single temporary buffer for A, B, and C)
    agm = (const __global realM*)((const __global real*)agm + a_offset);
    bgm = (const __global realN*)((const __global real*)bgm + b_offset);
    cgm = (__global realM*)((const __global real*)cgm + c_offset);
  
    // Allocates workgroup-private memory (local memory)
    #if SA == 1
        __local realM alm[KWG * MWG/VWM];
    #endif
    #if SB == 1
        __local realN blm[KWG * NWG/VWN];
    #endif
  
    // Computes the matrix-multiplication and stores the result in global memory
    #if SA == 1 && SB == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm,
          #ifdef BIAS
          egm,
          #endif
          cgm, alpha, beta, alm, blm);
    #elif SA == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm,
          #ifdef BIAS
          egm,
          #endif
          cgm, alpha, beta, alm);
    #elif SB == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm,
          #ifdef BIAS
          egm,
          #endif
          cgm, alpha, beta, blm);
    #else
        XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm,
          #ifdef BIAS
          egm,
          #endif
          cgm, alpha, beta);
    #endif
}

#if RELAX_WORKGROUP_SIZE == 1
    __kernel
#else
    __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void XgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                  const real_arg arg_alpha,
                  const real_arg arg_beta,
                  const __global realM* restrict agm, const int a_one, const int a_two,
                  const __global realN* restrict bgm, const int b_one, const int b_two,
                  __global realM* cgm, const int c_one, const int c_two) {
    const int batch = get_group_id(2);
    const real alpha = GetRealArg(arg_alpha);
    const real beta = GetRealArg(arg_beta);
    
    // Sets the offsets
    const int a_offset = batch * a_one * a_two;
    const int b_offset = batch * b_one * b_two;
    const int c_offset = batch * c_one * c_two;
    const __global realM* restrict agm_ = &agm[a_offset / VWM];
    const __global realN* restrict bgm_ = &bgm[b_offset / VWN];
    __global realM* restrict cgm_ = &cgm[c_offset / VWM];
  
    // Allocates workgroup-private memory (local memory)
    #if SA == 1
        __local realM alm[KWG * MWG/VWM];
    #endif
    #if SB == 1
        __local realN blm[KWG * NWG/VWN];
    #endif
  
    // Computes the matrix-multiplication and stores the result in global memory
    #if SA == 1 && SB == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm, blm);
    #elif SA == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm);
    #elif SB == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, blm);
    #else
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta);
    #endif
}
