#include <metal_stdlib>
using namespace metal;

// –––––––––––––––––––––––––––––––––––––––––––––––––––
// Macro
// –––––––––––––––––––––––––––––––––––––––––––––––––––

#define UP_DIV(x, y)    ( ((x) + (y) - 1) / (y) )
#define ROUND_UP(x, y)  ( ((x) + (y) - 1) / (y) * (y) )

// whether computer with float32 when store with float16
#define MNN_METAL_FLOAT32_COMPUTER 1 //

#if MNN_METAL_FULL_PRECISION
typedef float    ftype;
typedef float2   ftype2;
typedef float3   ftype3;
typedef float4   ftype4;
typedef float2x2 ftype2x2;
typedef float2x3 ftype2x3;
typedef float2x4 ftype2x4;
typedef float3x2 ftype3x2;
typedef float3x3 ftype3x3;
typedef float3x4 ftype3x4;
typedef float4x2 ftype4x2;
typedef float4x3 ftype4x3;
typedef float4x4 ftype4x4;
#else
typedef half     ftype;
typedef half2    ftype2;
typedef half3    ftype3;
typedef half4    ftype4;
typedef half2x2  ftype2x2;
typedef half2x3  ftype2x3;
typedef half2x4  ftype2x4;
typedef half3x2  ftype3x2;
typedef half3x3  ftype3x3;
typedef half3x4  ftype3x4;
typedef half4x2  ftype4x2;
typedef half4x3  ftype4x3;
typedef half4x4  ftype4x4;
#endif

#if MNN_METAL_FLOAT32_COMPUTER
typedef float    FLOAT;
typedef float2   FLOAT2;
typedef float3   FLOAT3;
typedef float4   FLOAT4;
typedef float2x2 FLOAT2x2;
typedef float2x3 FLOAT2x3;
typedef float2x4 FLOAT2x4;
typedef float3x2 FLOAT3x2;
typedef float3x3 FLOAT3x3;
typedef float3x4 FLOAT3x4;
typedef float4x2 FLOAT4x2;
typedef float4x3 FLOAT4x3;
typedef float4x4 FLOAT4x4;
#else
typedef half     FLOAT;
typedef half2    FLOAT2;
typedef half3    FLOAT3;
typedef half4    FLOAT4;
typedef half2x2  FLOAT2x2;
typedef half2x3  FLOAT2x3;
typedef half2x4  FLOAT2x4;
typedef half3x2  FLOAT3x2;
typedef half3x3  FLOAT3x3;
typedef half3x4  FLOAT3x4;
typedef half4x2  FLOAT4x2;
typedef half4x3  FLOAT4x3;
typedef half4x4  FLOAT4x4;
#endif

namespace MNN {
    
    // –––––––––––––––––––––––––––––––––––––––––––––––––––
    // Number Limit
    // –––––––––––––––––––––––––––––––––––––––––––––––––––
#define INT8_MAX    127
#define INT8_MIN    -128
#define INT16_MAX   32767
#define INT16_MIN   -32768
#define INT32_MAX   2147483647
#define INT32_MIN   -2147483648
#define UINT8_MAX   255
#define UINT16_MAX  65535
#define UINT32_MAX  4294967295U
    
    template<typename T> struct num_limits {
        static int max() { return 0; };
        static int min() { return 0; };
    };
    template<> struct num_limits<char> {
        static int max() { return INT8_MAX; };
        static int min() { return INT8_MIN; };
    };
    template<> struct num_limits<uchar> {
        static int max() { return UINT8_MAX; };
        static int min() { return 0; };
    };
    template<> struct num_limits<short> {
        static int max() { return INT16_MAX; };
        static int min() { return INT16_MIN; };
    };
    template<> struct num_limits<ushort> {
        static int max() { return UINT16_MAX; };
        static int min() { return 0; };
    };
    template<> struct num_limits<int> {
        static int max() { return INT32_MAX; };
        static int min() { return INT32_MIN; };
    };
    template<> struct num_limits<uint> {
        static int max() { return UINT32_MAX; };
        static int min() { return 0; };
    };
    
    // –––––––––––––––––––––––––––––––––––––––––––––––––––
    // Function
    // –––––––––––––––––––––––––––––––––––––––––––––––––––
    inline int dot(int4 i4, int4 w4) {
        return i4[0] * w4[0] + i4[1] * w4[1] + i4[2] * w4[2] + i4[3] * w4[3];
    }
    
    template <typename T>
    inline T saturate_round_x2_high_mul(T a, int b) {
        return mulhi(a, b) * 2;
    }
    
    template <typename T>
    inline T round_divide_by_pot(T x, int exponent) {
        int mask = (1 << exponent) - 1;
        T remainder = x & mask;
        T threshold = (mask >> 1) + T(x < 0);
        return (x >> exponent) + T(remainder > threshold);
    }
    
    // –––––––––––––––––––––––––––––––––––––––––––––––––––
    // Typedef
    // –––––––––––––––––––––––––––––––––––––––––––––––––––
    
    typedef struct short4x4 {
    private:
        short4 v[4];
    public:
        short4x4(short4 a) {
            v[0] = a; v[1] = a; v[2] = a; v[3] = a;
        }
        short4x4(short4 a, short4 b, short4 c, short4 d) {
            v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        }
        
        inline thread short4& operator[] (const int index) {
            return v[index];
        }
        inline device short4& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup short4& operator[] (const int index) threadgroup {
            return v[index];
        }
        
        inline const thread short4& operator[] (const int index) const {
            return v[index];
        }
        inline const device short4& operator[] (const int index) const device {
            return v[index];
        }
        inline const threadgroup short4& operator[] (const int index) const threadgroup {
            return v[index];
        }
        
        inline explicit operator half4x4() const {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        inline explicit operator half4x4() const device{
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        inline explicit operator half4x4() const threadgroup {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        
        inline explicit operator float4x4() const {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
        inline explicit operator float4x4() const device {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
        inline explicit operator float4x4() const threadgroup {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
    } short4x4;
    
    typedef struct char4x4 {
    private:
        char4 v[4];
    public:
        char4x4(char4 a) {
            v[0] = a; v[1] = a; v[2] = a; v[3] = a;
        }
        char4x4(char4 a, char4 b, char4 c, char4 d) {
            v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        }
        
        inline thread char4& operator[] (const int index) {
            return v[index];
        }
        inline device char4& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup char4& operator[] (const int index) threadgroup {
            return v[index];
        }
        
        inline const thread char4& operator[] (const int index) const {
            return v[index];
        }
        inline const device char4& operator[] (const int index) const device {
            return v[index];
        }
        inline const threadgroup char4& operator[] (const int index) const threadgroup {
            return v[index];
        }
        
        inline explicit operator half4x4() const {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        inline explicit operator half4x4() const device {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        inline explicit operator half4x4() const threadgroup {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        
        inline explicit operator float4x4() const {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
        inline explicit operator float4x4() const device {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
        inline explicit operator float4x4() const threadgroup {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
    } char4x4;

    typedef struct char4x2 {
    private:
        char2 v[4];
    public:
        char4x2(char2 a) {
            v[0] = a; v[1] = a; v[2] = a; v[3] = a;
        }
        char4x2(char2 a, char2 b, char2 c, char2 d) {
            v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        }
        
        inline thread char2& operator[] (const int index) {
            return v[index];
        }
        inline device char2& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup char2& operator[] (const int index) threadgroup {
            return v[index];
        }
        
        inline const thread char2& operator[] (const int index) const {
            return v[index];
        }
        inline const device char2& operator[] (const int index) const device {
            return v[index];
        }
        inline const threadgroup char2& operator[] (const int index) const threadgroup {
            return v[index];
        }
        
        inline explicit operator half4x2() const {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        inline explicit operator half4x2() const device {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        inline explicit operator half4x2() const threadgroup {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        
        inline explicit operator float4x2() const {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
        inline explicit operator float4x2() const device {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
        inline explicit operator float4x2() const threadgroup {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
    } char4x2;

    typedef struct uchar4x2 {
    private:
        uchar2 v[4];
    public:
        uchar4x2(uchar2 a) {
            v[0] = a; v[1] = a; v[2] = a; v[3] = a;
        }
        uchar4x2(uchar2 a, uchar2 b, uchar2 c, uchar2 d) {
            v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        }
        
        inline thread uchar2& operator[] (const int index) {
            return v[index];
        }
        inline device uchar2& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup uchar2& operator[] (const int index) threadgroup {
            return v[index];
        }
        
        inline const thread uchar2& operator[] (const int index) const {
            return v[index];
        }
        inline const device uchar2& operator[] (const int index) const device {
            return v[index];
        }
        inline const threadgroup uchar2& operator[] (const int index) const threadgroup {
            return v[index];
        }
        
        inline explicit operator half4x2() const {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        inline explicit operator half4x2() const device {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        inline explicit operator half4x2() const threadgroup {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        
        inline explicit operator float4x2() const {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
        inline explicit operator float4x2() const device {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
        inline explicit operator float4x2() const threadgroup {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
    } uchar4x2;
}
