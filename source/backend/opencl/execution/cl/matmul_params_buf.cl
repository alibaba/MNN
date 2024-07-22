#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// =================================================================================================
#define USE_INLINE_KEYWORD 1

#ifndef MWG
  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
  #define KWG 16      // Tile-size in dimension K (e.g. 8, 16)
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
  #define KWI 2      // Unroll factor of the KWG loop (smaller or equal than KWG)
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

#define ZERO (FLOAT)0.0f
// Sets a variable to zero
#define SetToZero(a) a = ZERO
#define IsZero(a) (a == ZERO)
#define Multiply(c,a,b) c = a * b
#if USE_CL_MAD == 1
#define MultiplyAdd(c,a,b) c = mad(a, b, c)
#else
#define MultiplyAdd(c,a,b) c += a * b
#endif

#define AXPBY(e,a,b,c,d) e = a*b + c*d

// Force inlining functions or not: some compilers don't support the inline keyword
#ifdef USE_INLINE_KEYWORD
  #define INLINE_FUNC inline
#else
  #define INLINE_FUNC
#endif


INLINE_FUNC int GetGroupID1() { return get_group_id(1); }
INLINE_FUNC int GetGroupID0() { return get_group_id(0); }

// =================================================================================================

// End of the C++11 raw string literal

typedef float real_arg;
#define GetRealArg(x) (FLOAT)x
typedef FLOAT real;

// Data-widths in dimension M
#if VWM == 1
    typedef FLOAT realM;
#elif VWM == 2
    typedef FLOAT2 realM;
#elif VWM == 4
    typedef FLOAT4 realM;
#elif VWM == 8
    typedef FLOAT8 realM;
#elif VWM == 16
    typedef FLOAT16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef FLOAT realN;
#elif VWN == 2
    typedef FLOAT2 realN;
#elif VWN == 4
    typedef FLOAT4 realN;
#elif VWN == 8
    typedef FLOAT8 realN;
#elif VWN == 16
    typedef FLOAT16 realN;
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
#if SA == 0
INLINE_FUNC int GlobalIndexA() {
  // Computes the indices based on strided/non-strided access
  #if STRM == 0
    // [MWG/MWI, MWI/VWM, VWM]
    int mg = get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    // [MWI/VWM, MWG/MWI, VWM]
    int mg = get_local_id(0);
  #endif

  // Computes the indices for the global memory
  // [kSizeM/MWG, (MWG/VWM), VWM]
  int idm = mg + GetGroupID0() * (MWG/VWM);
  return idm;
}

INLINE_FUNC realM GlobalToPrivateOptA(const __global realM* restrict agm, const int base, const int _mi,
                                   const int kSizeM, const int idk) {
  // Computes the indices based on strided/non-strided access
  #if STRM == 0
    // [MWG/MWI, MWI/VWM, VWM]
    int idm = base + _mi;
  #elif STRM == 1
    // [MWI/VWM, MWG/MWI, VWM]
    int idm = base + _mi*MDIMC;
  #endif

  // Loads the data from global memory (not transposed) and stores into registers
  // [kSizeK, kSizeM/VWM, VWM]
  return agm[idk*(kSizeM/VWM) + idm];
}

INLINE_FUNC realM GlobalToPrivateA(const __global realM* restrict agm, const int _mi,
                                   const int kSizeM, const int idk) {
  // Computes the indices based on strided/non-strided access
  #if STRM == 0
    // [MWG/MWI, MWI/VWM, VWM]
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    // [MWI/VWM, MWG/MWI, VWM]
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif

  // Computes the indices for the global memory
  // [kSizeM/MWG, (MWG/VWM), VWM]
  int idm = mg + GetGroupID0() * (MWG/VWM);

  // Loads the data from global memory (not transposed) and stores into registers
  // [kSizeK, kSizeM/VWM, VWM]
  return agm[idk*(kSizeM/VWM) + idm];
}

#endif

// Same as above, but now for the B input matrix
#if SB == 0
INLINE_FUNC int GlobalIndexB() {
  // Computes the indices based on strided/non-strided access
  #if STRN == 0
    int ng = get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1);
  #endif

  // Computes the indices for the global memory
  int idn = ng + GetGroupID1() * (NWG/VWN);
  return idn;
}

INLINE_FUNC realN GlobalToPrivateOptB(const __global realN* restrict bgm, const int base, const int _ni,
                                   const int kSizeN, const int idk) {
  // Computes the indices based on strided/non-strided access
  #if STRN == 0
  int idn = base + _ni;
  #elif STRN == 1
  int idn = base + _ni*NDIMC;
  #endif

  // Loads the data from global memory (transposed) and stores into registers
  return bgm[idk*(kSizeN/VWN) + idn];
}

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
    #if USE_CL_MAD == 1
    cvec = mad(avec, (realM)bval, cvec);
    #else
    cvec += avec * bval;
    #endif
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
    #if USE_CL_MAD == 1
    cvec = mad((realN)avec, bval, cvec);
    #else
    cvec += avec * bval;
    #endif
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

typedef struct {
    int index[2];
} INT2;

INLINE_FUNC INT2 StoreIndexM() {
  INT2 res;
  #if STRM == 0
    int mg = get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0);
  #endif
  #if STRN == 0
    int ng = get_local_id(1)*NWI;
  #elif STRN == 1
    int ng = get_local_id(1)*VWN;
  #endif
  int idm = mg + GetGroupID0() * (MWG/VWM);
  int idn = ng + GetGroupID1() * NWG;
  res.index[0] = idm;
  res.index[1] = idn;
  return res;
}

// layout : [N, M]
INLINE_FUNC void StoreResultsM(__global realM* cgm, realM c_value, const INT2 baseOffset, const int _mi, const int _ni,
                              const int kSizeM, const real alpha, const real beta) {
  #if STRM == 0
    int idm = _mi + baseOffset.index[0];
  #elif STRM == 1
    int idm = baseOffset.index[0] + _mi*MDIMC;
  #endif
  #if STRN == 0
    int idn = _ni + baseOffset.index[1];
  #elif STRN == 1
    int idn = _ni%VWN + baseOffset.index[1] + (_ni/VWN)*VWN*NDIMC;
  #endif
  
  int index = idn*(kSizeM/VWM) + idm;

  realM result = c_value;

  // The final multiplication with alpha (in case beta == 0)
  #ifdef ONLY_HAVE_ALPHA
    realM xval = c_value;
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
  #endif

  // The final multiplication with alpha and the addition with beta*C
  #ifdef HAVE_ALPHA_BETA
    realM xval = c_value;
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
  #endif
  cgm[index] = result;
}

INLINE_FUNC INT2 StoreIndexN() {
    INT2 res;
    #if STRM == 0
      int mg = get_local_id(0)*MWI;
    #elif STRM == 1
      int mg = get_local_id(0)*VWM;
    #endif
    #if STRN == 0
      int ng = get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1);
    #endif
    int idm = mg + GetGroupID0() * MWG;
    int idn = ng + GetGroupID1() * (NWG/VWN);
    
    res.index[0] = idm;
    res.index[1] = idn;
    return res;
}
// layout : [M, N]
INLINE_FUNC void StoreResultsN(__global realN* cgn, realN c_value,
                            const INT2 baseOffset,
                            #ifdef BIAS
                            realN* epm,
                            #endif
                            const int _mi, const int _ni,
                            const int kSizeN, const real alpha, const real beta) {

  #if STRM == 0
    int idm = _mi + baseOffset.index[0];
  #elif STRM == 1
    int idm = _mi%VWM + baseOffset.index[0] + (_mi/VWM)*VWM*MDIMC;
  #endif
  #if STRN == 0
    int idn = _ni + baseOffset.index[1];
  #elif STRN == 1
    int idn = baseOffset.index[1] + _ni*NDIMC;
  #endif

  int index = idm * (kSizeN/VWN) + idn;

  realN result = c_value;
  
  // The final multiplication with alpha (in case beta == 0)
  #ifdef ONLY_HAVE_ALPHA
    realN xval = c_value;
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
  #endif

  // The final multiplication with alpha and the addition with beta*C
  #ifdef HAVE_ALPHA_BETA
    realN xval = c_value;
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
  #endif
  
  
#ifdef BIAS
  realN eval = epm[_ni];

  #if VWN == 1
    result += eval;
    #ifdef RELU
    result = fmax(result, (FLOAT)0);
    #endif
    #ifdef RELU6
    result = clamp(result, (FLOAT)0, (FLOAT)6);
    #endif
  #elif VWN == 2
    result.x += eval.x;
    result.y += eval.y;
    #ifdef RELU
    result = fmax(result, (FLOAT2)0);
    #endif
    #ifdef RELU6
    result = clamp(result, (FLOAT2)0, (FLOAT2)6);
    #endif
  #elif VWN == 4
    result.x += eval.x;
    result.y += eval.y;
    result.z += eval.z;
    result.w += eval.w;
    #ifdef RELU
    result = fmax(result, (FLOAT4)0);
    #endif
    #ifdef RELU6
    result = clamp(result, (FLOAT4)0, (FLOAT4)6);
    #endif
  #elif VWN == 8
    result.s0 += eval.s0;
    result.s1 += eval.s1;
    result.s2 += eval.s2;
    result.s3 += eval.s3;
    result.s4 += eval.s4;
    result.s5 += eval.s5;
    result.s6 += eval.s6;
    result.s7 += eval.s7;
    #ifdef RELU
    result = fmax(result, (FLOAT8)0);
    #endif
    #ifdef RELU6
    result = clamp(result, (FLOAT8)0, (FLOAT8)6);
    #endif
  #elif VWN == 16
    result.s0 += eval.s0;
    result.s1 += eval.s1;
    result.s2 += eval.s2;
    result.s3 += eval.s3;
    result.s4 += eval.s4;
    result.s5 += eval.s5;
    result.s6 += eval.s6;
    result.s7 += eval.s7;
    result.s8 += eval.s8;
    result.s9 += eval.s9;
    result.sA += eval.sA;
    result.sB += eval.sB;
    result.sC += eval.sC;
    result.sD += eval.sD;
    result.sE += eval.sE;
    result.sF += eval.sF;
    #ifdef RELU
    result = fmax(result, (FLOAT16)0);
    #endif
    #ifdef RELU6
    result = clamp(result, (FLOAT16)0, (FLOAT16)6);
    #endif
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
  #if SA == 1 || SB == 1
      // Allocates workitem-private memory (registers)
      #pragma promote_to_registers
      realM apm[MWI/VWM]; // MWI * 1
      #pragma promote_to_registers
      realN bpm[NWI/VWN]; // 1 * NWI
      
      for (int kwg = 0; kwg < kSizeK; kwg += KWG) {
        // Loads data: off-chip --> local (matrix A)
        #if SA == 1
          GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
        #endif
        // Loads data: off-chip --> local (matrix B)
        #if SB == 1
          GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loops over all workitem tiles, unrolled by a factor KWI
        for (int pwi = 0; pwi < KWG; pwi += KWI) {
          #pragma unroll
          for (int _pit = 0; _pit < KWI; _pit += 1) {
            #if SA == 0 || SB == 0
              int idk = kwg + pwi + _pit;
            #endif
            int kg = pwi + _pit;

            // Loads matrix A (kernel 0) or matrix B (kernel 1)
            #pragma unroll
            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
              // Loads data: local --> private (matrix A)
              #if SA == 1
                apm[_mi] = LocalToPrivateA(alm, _mi, kg);
              // Loads data: off-chip --> private (matrix A)
              #elif SA == 0
                apm[_mi] = GlobalToPrivateA(agm, _mi, kSizeM, idk);
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
        barrier(CLK_LOCAL_MEM_FENCE);
      }
  #else
      // Allocates workitem-private memory (registers)

      int baseIndexA = GlobalIndexA();
      int baseIndexB = GlobalIndexB();

      #pragma unroll
      for (int _kj = 0; _kj < kSizeK; _kj += 4) {
        #ifdef OUTPUTMN
          #pragma promote_to_registers
          realN bpm[NWI/VWN]; // 1 * NWI
        
          #pragma unroll
          for(int _ki = 0; _ki < 4; _ki += 1) {
            int idk = _kj + _ki;
            #pragma unroll
            for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
              // Loads data: off-chip --> private (matrix B)
              bpm[_ni] = GlobalToPrivateOptB(bgm, baseIndexB, _ni, kSizeN, idk);
            }

            #pragma unroll
            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
              const realM aval = GlobalToPrivateOptA(agm, baseIndexA, _mi, kSizeM, idk);
              #pragma unroll
              for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
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
          }
        #else
        
          #pragma promote_to_registers
          realM apm[MWI/VWM]; // MWI * 1
          #pragma unroll
          for(int _ki = 0; _ki < 4; _ki += 1) {
            int idk = _kj + _ki;
            #pragma unroll
            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
              // Loads data: off-chip --> private (matrix B)
              apm[_mi] = GlobalToPrivateOptA(agm, baseIndexA, _mi, kSizeM, idk);
            }
            #pragma unroll
            for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
              const realN bval = GlobalToPrivateOptB(bgm, baseIndexB, _ni, kSizeN, idk);

              #pragma unroll
              for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
                const realM aval = apm[_mi];
                #if VWN == 1
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bval);
                #elif VWN == 2
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bval.x);
                  cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bval.y);
                #elif VWN == 4
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bval.x);
                  cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bval.y);
                  cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bval.z);
                  cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bval.w);
                #elif VWN == 8
                  cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bval.s0);
                  cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bval.s1);
                  cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bval.s2);
                  cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bval.s3);
                  cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bval.s4);
                  cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bval.s5);
                  cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bval.s6);
                  cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bval.s7);
                #elif VWN == 16
                  cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi], aval, bval.s0);
                  cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi], aval, bval.s1);
                  cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi], aval, bval.s2);
                  cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi], aval, bval.s3);
                  cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi], aval, bval.s4);
                  cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi], aval, bval.s5);
                  cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi], aval, bval.s6);
                  cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi], aval, bval.s7);
                  cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi], aval, bval.s8);
                  cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi], aval, bval.s9);
                  cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi], aval, bval.sA);
                  cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi], aval, bval.sB);
                  cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi], aval, bval.sC);
                  cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi], aval, bval.sD);
                  cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi], aval, bval.sE);
                  cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi], aval, bval.sF);
                #endif
              }
            }
          }
        #endif
      }
  #endif
  
  #if GLOBAL_MEM_FENCE == 1
    barrier(CLK_GLOBAL_MEM_FENCE);
  #endif

  #ifdef OUTPUTMN
      INT2 baseOffset = StoreIndexN();
      #ifdef BIAS
      #pragma promote_to_registers
      realN epm[NWI/VWN]; // MWI * 1
      for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          #if STRN == 0
            int idn = _ni + baseOffset.index[1];
          #elif STRN == 1
            int idn = baseOffset.index[1] + _ni*NDIMC;
          #endif
          epm[_ni] = egm[idn];
      }
      #endif
      #pragma unroll
      for (int _mi = 0; _mi < MWI; _mi += 1) {
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          StoreResultsN((__global realN* )cgm, cpn[_mi * (NWI/VWN) + _ni],
              baseOffset,
              #ifdef BIAS
              (realN*)epm,
              #endif
              _mi, _ni, kSizeN, alpha, beta);
        }
      }
  
  #else
      INT2 baseOffset = StoreIndexM();

      // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
      const int cld = kSizeM;
      
      #pragma unroll
      for (int _ni = 0; _ni < NWI; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
          StoreResultsM(cgm, cpm[_ni * (MWI/VWM) + _mi], baseOffset, _mi, _ni, cld, alpha, beta);
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
                  const __global realM* restrict agm, const int batch_offset_a,
                  const __global realN* restrict bgm, const int batch_offset_b,
                  #ifdef BIAS
                  const __global realN* restrict egm, const int batch_offset_e,
                  #endif
                  __global realM* cgm, const int batch_offset_c) {
    const int batch = get_group_id(2);
    const real alpha = GetRealArg(arg_alpha);
    const real beta = GetRealArg(arg_beta);
    
    // Sets the offsets
    const int a_offset = batch * batch_offset_a;
    const int b_offset = batch * batch_offset_b;
    const int c_offset = batch * batch_offset_c;
    const __global realM* restrict agm_ = &agm[a_offset / VWM];
    const __global realN* restrict bgm_ = &bgm[b_offset / VWN];
    __global realM* restrict cgm_ = &cgm[c_offset / VWM];
    
    #ifdef BIAS
    const int e_offset = batch * batch_offset_e;
    const __global realN* restrict egm_ = &egm[e_offset / VWN];
    #endif
  
    // Allocates workgroup-private memory (local memory)
    #if SA == 1
        __local realM alm[KWG * MWG/VWM];
    #endif
    #if SB == 1
        __local realN blm[KWG * NWG/VWN];
    #endif
  
    // Computes the matrix-multiplication and stores the result in global memory
    #if SA == 1 && SB == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_,
        #ifdef BIAS
        egm_,
        #endif
        cgm_, alpha, beta, alm, blm);
    #elif SA == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_,
        #ifdef BIAS
        egm_,
        #endif
        cgm_, alpha, beta, alm);
    #elif SB == 1
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_,
        #ifdef BIAS
        egm_,
        #endif
        cgm_, alpha, beta, blm);
    #else
        XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_,
        #ifdef BIAS
        egm_,
        #endif
        cgm_, alpha, beta);
    #endif
}
