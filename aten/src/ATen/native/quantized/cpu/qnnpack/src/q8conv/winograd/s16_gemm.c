#include <stdlib.h>
#include <arm_neon.h>

/*------------- Micro Kernel impl. ----------*/
void s16gemm_kernel(
        size_t mr, // mr <= 8
        size_t nr, // nr <= 8
        size_t k, // kc
        const int16_t* a,
        size_t a_stride,
        const int16_t* b,
        int32_t* c,
        size_t c_stride
        ){
    int32x4_t vacc0x0123 = vdupq_n_s32(0); // vld1q_s32(c);
    int32x4_t vacc0x4567 = vdupq_n_s32(0); // vld1q_s32(c+4);
    int32x4_t vacc1x0123 = vdupq_n_s32(0); // vld1q_s32(c+c_stride);
    int32x4_t vacc1x4567 = vdupq_n_s32(0); // vld1q_s32(c+c_stride+4);
    int32x4_t vacc2x0123 = vdupq_n_s32(0); // vld1q_s32(c+2*c_stride);
    int32x4_t vacc2x4567 = vdupq_n_s32(0); // vld1q_s32(c+2*c_stride+4);
    int32x4_t vacc3x0123 = vdupq_n_s32(0); // vld1q_s32(c+3*c_stride);
    int32x4_t vacc3x4567 = vdupq_n_s32(0); // vld1q_s32(c+3*c_stride+4);
    int32x4_t vacc4x0123 = vdupq_n_s32(0); // vld1q_s32(c+4*c_stride);
    int32x4_t vacc4x4567 = vdupq_n_s32(0); // vld1q_s32(c+4*c_stride+4);
    int32x4_t vacc5x0123 = vdupq_n_s32(0); // vld1q_s32(c+5*c_stride);
    int32x4_t vacc5x4567 = vdupq_n_s32(0); // vld1q_s32(c+5*c_stride+4);
    int32x4_t vacc6x0123 = vdupq_n_s32(0); // vld1q_s32(c+6*c_stride);
    int32x4_t vacc6x4567 = vdupq_n_s32(0); // vld1q_s32(c+6*c_stride+4);
    int32x4_t vacc7x0123 = vdupq_n_s32(0); // vld1q_s32(c+7*c_stride);
    int32x4_t vacc7x4567 = vdupq_n_s32(0); // vld1q_s32(c+7*c_stride+4);

    const int16_t* a0 = a;
    const int16_t* a1 = a0 + a_stride;
    if( mr < 2){
      a1 = a0;
    }
    const int16_t* a2 = a1 + a_stride;
    if( mr <= 2){
      a2 = a1;
    }
    const int16_t* a3 = a2 + a_stride;
    if( mr < 4){
      a3 = a2;
    }
    const int16_t* a4 = a3 + a_stride;
    if( mr <= 4){
      a4 = a3;
    }
    const int16_t* a5 = a4 + a_stride;
    if( mr < 6){
      a5 = a4;
    }
    const int16_t* a6 = a5 + a_stride;
    if( mr <= 6){
      a6 = a5;
    }
    const int16_t* a7 = a6 + a_stride;
    if( mr != 8){
      a7 = a6;
    }

    for(; k >=4 ; k-= 4){
        const int16x4_t va0 = vld1_s16(a0); a0 += 4;
        const int16x4_t va1 = vld1_s16(a1); a1 += 4;
        const int16x4_t va2 = vld1_s16(a2); a2 += 4;
        const int16x4_t va3 = vld1_s16(a3); a3 += 4;
        const int16x4_t va4 = vld1_s16(a4); a4 += 4;
        const int16x4_t va5 = vld1_s16(a5); a5 += 4;
        const int16x4_t va6 = vld1_s16(a6); a6 += 4;
        const int16x4_t va7 = vld1_s16(a7); a7 += 4;

        {
            const int16x8_t vb0 = vld1q_s16(b); b += 8;

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb0), va0, 0);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb0), va0, 0);

            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb0), va1, 0);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb0), va1, 0);

            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb0), va2, 0);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb0), va2, 0);

            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb0), va3, 0);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb0), va3, 0);

            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb0), va4, 0);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb0), va4, 0);

            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb0), va5, 0);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb0), va5, 0);

            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb0), va6, 0);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb0), va6, 0);

            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb0), va7, 0);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb0), va7, 0);
        }

        {
            const int16x8_t vb1 = vld1q_s16(b); b += 8;

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb1), va0, 1);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb1), va0, 1);

            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb1), va1, 1);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb1), va1, 1);

            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb1), va2, 1);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb1), va2, 1);

            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb1), va3, 1);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb1), va3, 1);

            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb1), va4, 1);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb1), va4, 1);

            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb1), va5, 1);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb1), va5, 1);

            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb1), va6, 1);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb1), va6, 1);

            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb1), va7, 1);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb1), va7, 1);
        }

        {
            const int16x8_t vb2 = vld1q_s16(b); b += 8;

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb2), va0, 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb2), va0, 2);

            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb2), va1, 2);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb2), va1, 2);

            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb2), va2, 2);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb2), va2, 2);

            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb2), va3, 2);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb2), va3, 2);

            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb2), va4, 2);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb2), va4, 2);

            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb2), va5, 2);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb2), va5, 2);

            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb2), va6, 2);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb2), va6, 2);

            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb2), va7, 2);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb2), va7, 2);
        }

        {
            const int16x8_t vb3 = vld1q_s16(b); b += 8;

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb3), va0, 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb3), va0, 3);

            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb3), va1, 3);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb3), va1, 3);

            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb3), va2, 3);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb3), va2, 3);

            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb3), va3, 3);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb3), va3, 3);

            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb3), va4, 3);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb3), va4, 3);

            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb3), va5, 3);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb3), va5, 3);

            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb3), va6, 3);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb3), va6, 3);

            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb3), va7, 3);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb3), va7, 3);
        }
    }

    if ( k != 0 ){
        const size_t a_precedence = 4 - k;
        const int64x1_t va_shift = vmov_n_s64(-16 * a_precedence);

        const int16x4_t va0 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a0 - a_precedence)), va_shift));
        const int16x4_t va1 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a1 - a_precedence)), va_shift));
        const int16x4_t va2 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a2 - a_precedence)), va_shift));
        const int16x4_t va3 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a3 - a_precedence)), va_shift));
        const int16x4_t va4 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a4 - a_precedence)), va_shift));
        const int16x4_t va5 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a5 - a_precedence)), va_shift));
        const int16x4_t va6 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a6 - a_precedence)), va_shift));
        const int16x4_t va7 = vreinterpret_s16_u64(vshl_u64(vreinterpret_u64_s16(vld1_s16(a7 - a_precedence)), va_shift));

        const int16x8_t vb0 = vld1q_s16(b); b += 8;
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb0), va0, 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb0), va0, 0);

        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb0), va1, 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb0), va1, 0);

        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb0), va2, 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb0), va2, 0);

        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb0), va3, 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb0), va3, 0);

        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb0), va4, 0);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb0), va4, 0);

        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb0), va5, 0);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb0), va5, 0);

        vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb0), va6, 0);
        vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb0), va6, 0);

        vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb0), va7, 0);
        vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb0), va7, 0);

        if( k >= 2){
            const int16x8_t vb1 = vld1q_s16(b); b += 8;
            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb1), va0, 1);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb1), va0, 1);

            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb1), va1, 1);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb1), va1, 1);

            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb1), va2, 1);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb1), va2, 1);

            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb1), va3, 1);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb1), va3, 1);

            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb1), va4, 1);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb1), va4, 1);

            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb1), va5, 1);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb1), va5, 1);

            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb1), va6, 1);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb1), va6, 1);

            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb1), va7, 1);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb1), va7, 1);
            if( k >= 3){
                const int16x8_t vb2 = vld1q_s16(b); b += 8;
                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb2), va0, 2);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb2), va0, 2);

                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb2), va1, 2);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb2), va1, 2);

                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb2), va2, 2);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb2), va2, 2);

                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb2), va3, 2);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb2), va3, 2);

                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vb2), va4, 2);
                vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vb2), va4, 2);

                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vb2), va5, 2);
                vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vb2), va5, 2);

                vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vb2), va6, 2);
                vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vb2), va6, 2);

                vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vb2), va7, 2);
                vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vb2), va7, 2);
            }
        }
    }

    int32_t* c0 = c;
    int32_t* c1 = c0 + c_stride;
    if( mr < 2){
        c1 = c0;
    }
    int32_t* c2 = c1 + c_stride;
    if ( mr <= 2) {
        c2 = c1;
    }
    int32_t* c3 = c2 + c_stride;
    if ( mr < 4){
        c3 = c2;
    }
    int32_t* c4 = c3 + c_stride;
    if ( mr <= 4){
        c4 = c3;
    }
    int32_t* c5 = c4 + c_stride;
    if ( mr < 6 ){
        c5 = c4;
    }
    int32_t* c6 = c5 + c_stride;
    if ( mr <= 6){
        c6 = c5;
    }
    int32_t* c7 = c6 + c_stride;
    if ( mr != 8){
        c7 = c6;
    }

    if( nr == 8){
        vst1q_s32(c0, vacc0x0123);
        vst1q_s32(c0+4, vacc0x4567);

        vst1q_s32(c1, vacc1x0123);
        vst1q_s32(c1+4, vacc1x4567);

        vst1q_s32(c2, vacc2x0123);
        vst1q_s32(c2+4, vacc2x4567);

        vst1q_s32(c3, vacc3x0123);
        vst1q_s32(c3+4, vacc3x4567);

        vst1q_s32(c4, vacc4x0123);
        vst1q_s32(c4+4, vacc4x4567);

        vst1q_s32(c5, vacc5x0123);
        vst1q_s32(c5+4, vacc5x4567);

        vst1q_s32(c6, vacc6x0123);
        vst1q_s32(c6+4, vacc6x4567);

        vst1q_s32(c7, vacc7x0123);
        vst1q_s32(c7+4, vacc7x4567);
    } 
    else {
        if( nr >= 4){
            vst1q_s32(c0, vacc0x0123); c0 += 4;
            vst1q_s32(c1, vacc1x0123); c1 += 4;
            vst1q_s32(c2, vacc2x0123); c2 += 4;
            vst1q_s32(c3, vacc3x0123); c3 += 4;
            vst1q_s32(c4, vacc4x0123); c4 += 4;
            vst1q_s32(c5, vacc5x0123); c5 += 4;
            vst1q_s32(c6, vacc6x0123); c6 += 4;
            vst1q_s32(c7, vacc7x0123); c7 += 4;

            vacc0x0123 = vacc0x4567;
            vacc1x0123 = vacc1x4567;
            vacc2x0123 = vacc2x4567;
            vacc3x0123 = vacc3x4567;
            vacc4x0123 = vacc4x4567;
            vacc5x0123 = vacc5x4567;
            vacc6x0123 = vacc6x4567;
            vacc7x0123 = vacc7x4567;
            nr -= 4;
        }
        if ( nr >= 2){
            vst1_s32(c0, vget_low_s32(vacc0x0123)); c0 += 2;
            vst1_s32(c1, vget_low_s32(vacc1x0123)); c1 += 2;
            vst1_s32(c2, vget_low_s32(vacc2x0123)); c2 += 2;
            vst1_s32(c3, vget_low_s32(vacc3x0123)); c3 += 2;
            vst1_s32(c4, vget_low_s32(vacc4x0123)); c4 += 2;
            vst1_s32(c5, vget_low_s32(vacc5x0123)); c5 += 2;
            vst1_s32(c6, vget_low_s32(vacc6x0123)); c6 += 2;
            vst1_s32(c7, vget_low_s32(vacc7x0123)); c7 += 2;

            vacc0x0123 = vextq_s32(vacc0x0123, vacc0x0123, 2);
            vacc1x0123 = vextq_s32(vacc1x0123, vacc1x0123, 2);
            vacc2x0123 = vextq_s32(vacc2x0123, vacc2x0123, 2);
            vacc3x0123 = vextq_s32(vacc3x0123, vacc3x0123, 2);
            vacc4x0123 = vextq_s32(vacc4x0123, vacc4x0123, 2);
            vacc5x0123 = vextq_s32(vacc5x0123, vacc5x0123, 2);
            vacc6x0123 = vextq_s32(vacc6x0123, vacc6x0123, 2);
            vacc7x0123 = vextq_s32(vacc7x0123, vacc7x0123, 2);
            nr -= 2;
        }
        if ( nr != 0){
            vst1q_lane_s32(c0, vacc0x0123, 0);
            vst1q_lane_s32(c1, vacc0x0123, 0);
            vst1q_lane_s32(c2, vacc0x0123, 0);
            vst1q_lane_s32(c3, vacc0x0123, 0);
            vst1q_lane_s32(c4, vacc0x0123, 0);
            vst1q_lane_s32(c5, vacc0x0123, 0);
            vst1q_lane_s32(c6, vacc0x0123, 0);
            vst1q_lane_s32(c7, vacc0x0123, 0);
        }
    }
}

void s16gemm(
        const int16_t* a,
        const int16_t* b,
        int32_t* c,
        size_t m,
        size_t n,
        size_t k
        )
{
    const size_t mr = 8;
    const size_t nr = 8;
    const size_t a_stride = k;
    const size_t c_stride = n;
    for(int jr = 0; jr < n; jr += nr){
        for(int ir=0; ir < m; ir += mr){
            size_t _mr = m - ir < mr ? m % mr : mr;
            size_t _nr = n - jr < nr ? n % nr : nr;
            s16gemm_kernel(
                    _mr, _nr, k,
                    a + ir * a_stride, a_stride,
                    b,
                    c + ir * c_stride + jr,
                    c_stride
                    );
        }
        b += k * nr;
    }
}

// void s16_gemm(
//         int16_t* a, // input: m x k
//         int16_t* b, // weight: k x n
//         int32_t* c, // output: m x n
//         const int m, // tile size
//         const int k, // input channels
//         const int n // output channels
//         ){
// 
//     /*--Cache Blocking Params---*/
//     const int mc = 1024;
//     const int kc = 64;
//     const int nc = 1024;
//     /*--Cache Blocking Params---*/
// 
//     /*--Register Blocking Params --*/
//     const int nr = 8;
//     const int mr = 8;
//     /*--Register Blocking Params --*/
//     for(int jc = 0; jc < n; jc += nc){
//         for(int pc = 0; pc < k; pc += kc){
//             // Bc = B[pc: pc+kc, jc:jc+nc]
//             int16_t* bc = b + pc*n + pc + kc;
//             for(int ic = 0; ic < m; ic += mc){
//                 // Ac = A[ic:ic+mc, pc:pc+kc]
//                 int16_t* ac = a + ic*k + ic+mc;
// /*----------------------- Macro Kernel ---------------------------*/
//                 for(int jr = 0; jr < nc; jr+=nr){
//                     for(int ir=0; ir < mc; ir+=mr){
//   /*--------------------- Micro Kernel -------------------------*/
//                         // Cc[ir:ir+mr, jr:jr+nr] =
//                         // Ac[ir:ir+mr, pr] *
//                         // Bc[pr, jr:jr+nr]
//                         s16_gemm_micro_kernel_mr_nr_8x8(
//                                 ac + ir*k,
//                                 bc + jr,
//                                 c + ir*m + jr,
//                                 k, n, n,
//                                 kc
//                                 );
//   /*--------------------- Micro Kernel -------------------------*/
//                     }
//                 }
// /*----------------------- Macro Kernel ---------------------------*/
//             }
//         }
//     }
// }
// 
