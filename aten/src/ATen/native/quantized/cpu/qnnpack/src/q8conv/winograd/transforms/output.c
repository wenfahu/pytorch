#include <arm_neon.h>
#include <stddef.h>
#include <pytorch_qnnpack.h>
#include "../winograd.h"


void winograd_f_2_3_output_transform(
        const int32_t* restrict input_base,
        const int32_t* restrict bias_ptr,
        const uint8_t* restrict output,
        const int n_channels,
        const int output_row_stride,
        const int output_col_stride,
        const int matrix_stride,
        const union pytorch_qnnp_conv_quantization_params quantization_params[restrict static 1]
		)
{
  const int output_tile_rows = 2, output_tile_cols = 2;
  // Construct a map to the output cells
  uint8_t *outptrs[output_tile_rows][output_tile_cols];
  for (int i = 0; i < output_tile_rows; i++)
  {
    for (int j = 0; j < output_tile_cols; j++)
    {
      outptrs[i][j] = output + i * output_row_stride + j * output_col_stride;
    }
  }

  // For each channel of the output
  int channels_remaining = n_channels;
  for (; channels_remaining >= 8; channels_remaining -= 8)
  {
      int32x4_t f[4][2];
      for (int c=0; c < 2; c++)
      {
        // Matrices used and computed during this transform
        int32x4_t F[4][4], FZ[4][2], b;

        // Read a 4x4 tile in the Winograd domain
        for (int i = 0, m = 0; i < 4; i++)
        {
          for (int j = 0; j < 4; j++, m++)
          {
            F[i][j] = vld1q_s32(input_base + m*matrix_stride);
          }
        }
        input_base += 4;

        // Compute the matrix F Z
        for (int i = 0; i < 4; i++)
        {
          // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
          FZ[i][0] = vaddq_s32(vaddq_s32(F[i][0], F[i][1]), F[i][2]);

          // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
          FZ[i][1] = vsubq_s32(vsubq_s32(F[i][1], F[i][2]), F[i][3]);
        }

        // Compute the output tile f = ZT F Z
        for (int j = 0; j < 2; j++)
        {
          // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
          f[0+c*2][j] = vaddq_s32(vaddq_s32(FZ[0][j], FZ[1][j]), FZ[2][j]);

          // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
          f[1+c*2][j] = vsubq_s32(vsubq_s32(FZ[1][j], FZ[2][j]), FZ[3][j]);
        }
      }

      // DEBUG ONLY
      // for(int i=0; i<4; i++){
      //     for(int j=0; j<2; j++){
      //         for(int k=0;k<4;k++){
      //             printf("%d\t", f[i][j][k]);
      //         }
      //     }
      // }


      int32x4_t vbx0123, vbx4567;

      if (bias_ptr != NULL)
      {
          vbx0123 = vld1q_s32(bias_ptr); bias_ptr += 4;
          vbx4567 = vld1q_s32(bias_ptr); bias_ptr += 4;
      }
      else
      {
          vbx0123 = vdupq_n_s32(0);
          vbx4567 = vdupq_n_s32(0);
      }

      f[0][0] = vaddq_s32(vshrq_n_s32(f[0][0], 2), vbx0123);
      f[0][1] = vaddq_s32(vshrq_n_s32(f[0][1], 2), vbx0123);
      f[1][0] = vaddq_s32(vshrq_n_s32(f[1][0], 2), vbx0123);
      f[1][1] = vaddq_s32(vshrq_n_s32(f[1][1], 2), vbx0123);
      f[2][0] = vaddq_s32(vshrq_n_s32(f[2][0], 2), vbx4567);
      f[2][1] = vaddq_s32(vshrq_n_s32(f[2][1], 2), vbx4567);
      f[3][0] = vaddq_s32(vshrq_n_s32(f[3][0], 2), vbx4567);
      f[3][1] = vaddq_s32(vshrq_n_s32(f[3][1], 2), vbx4567);


      // concat f[0:2, :] and f[@:, :] so each tile postion has 8 elements
    const int32x4_t vmultiplier = vld1q_dup_s32(&quantization_params->neon.multiplier);
    const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    for(int i=0; i< 4; i++){
        for(int j = 0; j < 2; j++){
            f[i][j] = vqrdmulhq_s32(f[i][j], vmultiplier);
            f[i][j] = vsraq_n_s32(f[i][j], vbicq_s32(f[i][j], vzero_shift_mask), 31);
            f[i][j] = vrshlq_s32(f[i][j], vright_shift);
        }
    }
    const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);

    int16x8_t qf[2][2];
    qf[0][0] = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(f[0][0]), f[2][0]), voutput_zero_point);
    qf[0][1] = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(f[0][1]), f[2][1]), voutput_zero_point);
    qf[1][0] = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(f[1][0]), f[3][0]), voutput_zero_point);
    qf[1][1] = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(f[1][1]), f[3][1]), voutput_zero_point);

    uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(qf[0][0]), qf[0][1]);
    uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(qf[1][0]), qf[1][1]);

    const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
    vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);

    vst1_u8(outptrs[0][0], vget_low_u8(vout0x01234567_1x01234567));  outptrs[0][0] += 8;
    vst1_u8(outptrs[0][1], vget_high_u8(vout0x01234567_1x01234567)); outptrs[0][1] += 8;
    vst1_u8(outptrs[1][0], vget_low_u8(vout2x01234567_3x01234567));  outptrs[1][0] += 8;
    vst1_u8(outptrs[1][1], vget_high_u8(vout2x01234567_3x01234567)); outptrs[1][1] += 8;
  }

  if ( channels_remaining != 0){

      if (channels_remaining >= 4){
        // Transform 4 elements
        // Matrices used and computed during this transform
        int32x4_t F[4][4], FZ[4][2], f[2][2] ;

        // Read a 4x4 tile in the Winograd domain
        for (int i = 0, m = 0; i < 4; i++)
        {
          for (int j = 0; j < 4; j++, m++)
          {
            F[i][j] = vld1q_s32(input_base + m*matrix_stride);
          }
        }
        input_base += 4;

        // Compute the matrix F Z
        for (int i = 0; i < 4; i++)
        {
          // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
          FZ[i][0] = vaddq_s32(vaddq_s32(F[i][0], F[i][1]), F[i][2]);

          // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
          FZ[i][1] = vsubq_s32(vsubq_s32(F[i][1], F[i][2]), F[i][3]);
        }

        // Compute the output tile f = ZT F Z
        for (int j = 0; j < 2; j++)
        {
          // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
          f[0][j] = vaddq_s32(vaddq_s32(FZ[0][j], FZ[1][j]), FZ[2][j]);

          // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
          f[1][j] = vsubq_s32(vsubq_s32(FZ[1][j], FZ[2][j]), FZ[3][j]);
        }

        int32x4_t vbx0123;
        if (bias_ptr != NULL)
        {
            vbx0123 = vld1q_s32(bias_ptr); bias_ptr += 4;
        }
        else 
        {
            vbx0123 = vdupq_n_s32(0);
        }

        const int32x4_t vmultiplier = vld1q_dup_s32(&quantization_params->neon.multiplier);
        const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);
        const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
        for(int i=0; i< 2; i++){
            for(int j = 0; j < 2; j++){
                f[i][j] = vaddq_s32(vshrq_n_s32(f[i][j], 2), vbx0123);
                f[i][j] = vqrdmulhq_s32(f[i][j], vmultiplier);
                f[i][j] = vsraq_n_s32(f[i][j], vbicq_s32(f[i][j], vzero_shift_mask), 31);
                f[i][j] = vrshlq_s32(f[i][j], vright_shift);
            }
        }
        const int32x4_t voutput_zero_point = vmovl_s16(vld1_dup_s16(&quantization_params->neon.output_zero_point));

        int32x4_t vout00x1234 = vqaddq_s32(f[0][0], voutput_zero_point);
        int32x4_t vout01x1234 = vqaddq_s32(f[0][1], voutput_zero_point);
        int32x4_t vout10x1234 = vqaddq_s32(f[1][0], voutput_zero_point);
        int32x4_t vout11x1234 = vqaddq_s32(f[1][1], voutput_zero_point);

        int16x8_t vout00x1234_01x1234 = vqmovn_high_s32(vqmovn_s32(vout00x1234),
                vout01x1234);
        int16x8_t vout10x1234_11x1234 = vqmovn_high_s32(vqmovn_s32(vout10x1234),
                vout11x1234);

        uint8x16_t vout00x1234_01x1234_10x1234_11x1234 = vqmovun_high_s16(
                vqmovun_s16(vout00x1234_01x1234), vout10x1234_11x1234);

        const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
        const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);

        vout00x1234_01x1234_10x1234_11x1234 = vmaxq_u8(vout00x1234_01x1234_10x1234_11x1234, voutput_min);
        vout00x1234_01x1234_10x1234_11x1234 = vminq_u8(vout00x1234_01x1234_10x1234_11x1234, voutput_max);

        vst1q_lane_u32(outptrs[0][0], vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 0); outptrs[0][0] += 4;
        vst1q_lane_u32(outptrs[0][1], vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 1); outptrs[0][1] += 4;
        vst1q_lane_u32(outptrs[1][0], vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 2); outptrs[1][0] += 4;
        vst1q_lane_u32(outptrs[1][1], vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 3); outptrs[1][1] += 4;

        channels_remaining -= 4;
      }

      if ( channels_remaining != 0 ) {

          // overlap to 4
          // Transform 4 elements
        const predecrement = 4 - channels_remaining;
            // Matrices used and computed during this transform
        int32x4_t F[4][4], FZ[4][2], f[2][2];

        // Read a 4x4 tile in the Winograd domain
        for (int i = 0, m = 0; i < 4; i++)
        {
          for (int j = 0; j < 4; j++, m++)
          {
            F[i][j] = vld1q_s32(input_base + m*matrix_stride - predecrement);
          }
        }
        input_base += 4;

        // Compute the matrix F Z
        for (int i = 0; i < 4; i++)
        {
          // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
          FZ[i][0] = vaddq_s32(vaddq_s32(F[i][0], F[i][1]), F[i][2]);

          // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
          FZ[i][1] = vsubq_s32(vsubq_s32(F[i][1], F[i][2]), F[i][3]);
        }

        // Compute the output tile f = ZT F Z
        for (int j = 0; j < 2; j++)
        {
          // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
          f[0][j] = vaddq_s32(vaddq_s32(FZ[0][j], FZ[1][j]), FZ[2][j]);

          // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
          f[1][j] = vsubq_s32(vsubq_s32(FZ[1][j], FZ[2][j]), FZ[3][j]);
        }
          
        int32x4_t vbx0123;
        if (bias_ptr != NULL)
        {
            vbx0123 = vld1q_s32(bias_ptr - predecrement);
        }
        else 
        {
            vbx0123 = vdupq_n_s32(0);
        }

        const int32x4_t vmultiplier = vld1q_dup_s32(&quantization_params->neon.multiplier);
        const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);
        const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
        for(int i=0; i< 2; i++){
            for(int j = 0; j < 2; j++){
                f[i][j] = vaddq_s32(vshrq_n_s32(f[i][j], 2), vbx0123);
                f[i][j] = vqrdmulhq_s32(f[i][j], vmultiplier);
                f[i][j] = vsraq_n_s32(f[i][j], vbicq_s32(f[i][j], vzero_shift_mask), 31);
                f[i][j] = vrshlq_s32(f[i][j], vright_shift);
            }
        }
        const int32x4_t voutput_zero_point = vmovl_s16(vld1_dup_s16(&quantization_params->neon.output_zero_point));

        int32x4_t vout00x1234 = vqaddq_s32(f[0][0], voutput_zero_point);
        int32x4_t vout01x1234 = vqaddq_s32(f[0][1], voutput_zero_point);
        int32x4_t vout10x1234 = vqaddq_s32(f[1][0], voutput_zero_point);
        int32x4_t vout11x1234 = vqaddq_s32(f[1][1], voutput_zero_point);

        int16x8_t vout00x1234_01x1234 = vqmovn_high_s32(vqmovn_s32(vout00x1234),
                vout01x1234);
        int16x8_t vout10x1234_11x1234 = vqmovn_high_s32(vqmovn_s32(vout10x1234),
                vout11x1234);

        uint8x16_t vout00x1234_01x1234_10x1234_11x1234 = vqmovun_high_s16(
                vqmovun_s16(vout00x1234_01x1234), vout10x1234_11x1234);

        const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
        const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);

        vout00x1234_01x1234_10x1234_11x1234 = vmaxq_u8(vout00x1234_01x1234_10x1234_11x1234, voutput_min);
        vout00x1234_01x1234_10x1234_11x1234 = vminq_u8(vout00x1234_01x1234_10x1234_11x1234, voutput_max);

        // overlap store
        vst1q_lane_u32(outptrs[0][0] - predecrement, vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 0); outptrs[0][0] += 4;
        vst1q_lane_u32(outptrs[0][1] - predecrement, vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 1); outptrs[0][1] += 4;
        vst1q_lane_u32(outptrs[1][0] - predecrement, vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 2); outptrs[1][0] += 4;
        vst1q_lane_u32(outptrs[1][1] - predecrement, vreinterpretq_u32_u8(vout00x1234_01x1234_10x1234_11x1234), 3); outptrs[1][1] += 4;

      }
    }
}
