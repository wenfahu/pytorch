#include <arm_neon.h>

// transpose M H W C to H W C M
// https://github.com/pytorch/QNNPACK/issues/45#issuecomment-457503557
static inline void reorder_q8conv_w(
        const uint8_t* a,
        uint8_t* c,
        const int M,
        const int C,
        const int kH,
        const int kW
        ){
    const int stride_m = C * kH * kW;
    const int stride_c = 1;
    const int stride_w = C;
    const int stride_h = kW * C;
    
    for(int i=0, n=0; i < kH; i++){
        for(int j=0; j < kW; j++){
            for(int k=0; k < C; k++){
                for(int m=0; m < M; m++, n++){
                    int idx = m * stride_m + k * stride_c + j * stride_w + i * stride_h;
                    c[n] = a[idx];
                }
            }
        }
    }
}
