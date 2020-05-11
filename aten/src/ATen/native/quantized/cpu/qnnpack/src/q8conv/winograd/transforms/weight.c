#include <arm_neon.h>
#include <stdlib.h>
#include "../winograd.h"

// output dim (16, CHIN, CHOUT)
void winograd_f_2_3_kernel_transform(
        const uint8_t* restrict input_base, // HWIO
        int16_t* restrict output,
        const int input_channels,
        const int output_channels,
        const int matrix_stride, // ic * oc
        const int matrix_row_stride, // oc
        const uint8_t kernel_zero_point
		){
    const uint8x8_t vb_zero_point = vdup_n_u8(kernel_zero_point);
    const int inner_tile_i = 4, inner_tile_j = 4;

    const int weight_col_stride = output_channels * input_channels;
    const int weight_row_stride = 3 * weight_col_stride;
    const uint8_t* inptrs[3][3];

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            inptrs[i][j] = input_base + i * weight_row_stride + j * weight_col_stride;
        }
    }
  // For each input channel
  for (int ic = 0; ic < input_channels; ic++) {
    int16_t *outptr = output + ic * matrix_row_stride;

    // For each output channel
    int channels_remaining = output_channels;
    for (; channels_remaining >= 8; channels_remaining -= 8) {
      // Matrices used and computed in this kernel
      int16x8_t w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

      // Read weights
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          const uint8x8_t vec = vld1_u8(inptrs[i][j]);
          w[i][j] = vreinterpretq_s16_u16(vsubl_u8(vec, vb_zero_point));
          inptrs[i][j] += 8;
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 3; j++) {
        // Ww[0][j] = 2 * w[0][j];
        Ww[0][j] = vmulq_n_s16(w[0][j], ((short)2));

        // Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);
        // Ww[1][j] = vmulq_n_f32(vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);
        Ww[1][j] = vaddq_s16(vaddq_s16(w[0][j], w[1][j]), w[2][j]);

        // Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);
        // Ww[2][j] = vmulq_n_f32(vaddq_f32(vsubq_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);
        Ww[2][j] = vaddq_s16(vsubq_s16(w[0][j], w[1][j]), w[2][j]);

        Ww[3][j] = vmulq_n_s16(w[2][j], ((short)2));
      }

      // Compute V = W w WT
      for (int i = 0; i < inner_tile_i; i++) {
        V[i][0] = vmulq_n_s16(Ww[i][0], ((short)2));

        // V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
        // V[i][1] = vmulq_n_f32(vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);
        V[i][1] = vaddq_s16(vaddq_s16(Ww[i][0], Ww[i][1]), Ww[i][2]);

        // V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
        // V[i][2] = vmulq_n_f32(vaddq_f32(vsubq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);
        V[i][2] = vaddq_s16(vsubq_s16(Ww[i][0], Ww[i][1]), Ww[i][2]);

        V[i][3] = vmulq_n_s16(Ww[i][2], ((short)2));
      }

      // Store the transformed weights: IHWO layout
      for (int i = 0, m = 0; i < inner_tile_i; i++) {
        for (int j = 0; j < inner_tile_j; j++, m++) {
          vst1q_s16(outptr + m*matrix_stride, V[i][j]);
        }
      }
      outptr += 8;
    }

     int16x8_t w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];
     if ( channels_remaining != 0){
       const size_t predecrement = 8 - channels_remaining;
       const int64x1_t v_shift = vmov_n_s64(-8 * predecrement);
       //
       // Read weights
       for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
           const uint8x8_t vec = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(inptrs[i][j] - predecrement)), v_shift));
           inptrs[i][j] += 4;
           w[i][j] = vreinterpretq_s16_u16(vsubl_u8(vec, vb_zero_point));
         }
       }
       //
       // Compute the matrix W w
       for (int j = 0; j < 3; j++) {
         // Ww[0][j] = 2 * w[0][j];
         Ww[0][j] = vmulq_n_s16(w[0][j], ((short)2));
 
         // Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);
         // Ww[1][j] = vmulq_n_f32(vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);
         Ww[1][j] = vaddq_s16(vaddq_s16(w[0][j], w[1][j]), w[2][j]);
 
         // Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);
         // Ww[2][j] = vmulq_n_f32(vaddq_f32(vsubq_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);
         Ww[2][j] = vaddq_s16(vsubq_s16(w[0][j], w[1][j]), w[2][j]);
 
         Ww[3][j] = vmulq_n_s16(w[2][j], ((short)2));
       }
 
       // Compute V = W w WT
       for (int i = 0; i < inner_tile_i; i++) {
         V[i][0] = vmulq_n_s16(Ww[i][0], ((short)2));
 
         // V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
         // V[i][1] = vmulq_n_f32(vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);
         V[i][1] = vaddq_s16(vaddq_s16(Ww[i][0], Ww[i][1]), Ww[i][2]);
 
         // V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
         // V[i][2] = vmulq_n_f32(vaddq_f32(vsubq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);
         V[i][2] = vaddq_s16(vsubq_s16(Ww[i][0], Ww[i][1]), Ww[i][2]);
 
         V[i][3] = vmulq_n_s16(Ww[i][2], ((short)2));
       }
 
     }
     if (channels_remaining >= 4){
       for (int i = 0, m = 0; i < inner_tile_i; i++) {
         for (int j = 0; j < inner_tile_j; j++, m++) {
           vst1_s16(outptr + m*matrix_stride, vget_low_s16(V[i][j]));
           V[i][j] = vextq_s16(V[i][j], V[i][j], 4);
         }
       }
       outptr += 4;
       channels_remaining -= 4;
     }
     if (channels_remaining >= 2){
       for (int i = 0, m = 0; i < inner_tile_i; i++) {
         for (int j = 0; j < inner_tile_j; j++, m++) {
           vst1q_lane_s32(outptr + m*matrix_stride, vreinterpretq_s32_s16(V[i][j]), 0);
           V[i][j] = vextq_s16(V[i][j], V[i][j], 2);
         }
       }
       outptr += 2;
       channels_remaining -= 2;
     }
     if (channels_remaining != 0){
       for (int i = 0, m = 0; i < inner_tile_i; i++) {
         for (int j = 0; j < inner_tile_j; j++, m++) {
           vst1q_lane_s16(outptr + m*matrix_stride, V[i][j], 0);
         }
       }
     }
   }
}
