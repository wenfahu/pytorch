#include <arm_neon.h>
#include <stdio.h>
#include "../winograd.h"

// output dim (16, W*H/4, CH)
void winograd_f_2_3_input_transform(
        const uint8_t* input_base,
        const int16_t* outptr,
        const int input_row_stride, // c * w
        const int input_col_stride, // c
        const int channels,
        const int matrix_stride, // num_tiles x c
        const uint8_t input_zero_point
		){

    const uint8x8_t va_zero_point = vdup_n_u8(input_zero_point);
    const int tile_rows = 4, tile_cols = 4;

    const uint8_t* ptrs[tile_rows][tile_cols];

    // Get pointers into the input tile
    for (int i = 0, xi = 0; i < tile_rows; i++, xi++) {
        // Get a pointer into the row
        const uint8_t* const row_ptr = input_base + xi*input_row_stride;

        for (int j = 0, xj = 0; j < tile_cols; j++, xj++) {
            ptrs[i][j] = row_ptr + xj * input_col_stride;
        }
    }

    int remaining_channels = channels;
    for(; remaining_channels >= 8; remaining_channels -= 8){

        // Winograd Input transform 4x4x4 tile
        // U = BT d B = (BT (BT d)T)T
        int16x8_t d[tile_rows][tile_cols];
        int16x8_t BTd[tile_rows][tile_cols];
        int16x8_t U[tile_rows][tile_cols];

        // load d
        for (int i = 0; i < tile_rows; i++)
        {
            for (int j = 0; j < tile_cols; j++)
            {
                const uint8x8_t vec = vld1_u8(ptrs[i][j]);
                d[i][j] = vreinterpretq_s16_u16(vsubl_u8(vec, va_zero_point));
                ptrs[i][j] += 8;
            }
        }

        // BT d
        for (int j = 0; j < tile_cols; j++)
        {
          // BTd[0][j] = d[0][j] - d[2][j];
          BTd[0][j] = vsubq_s16(d[0][j], d[2][j]);

          // BTd[1][j] = d[1][j] + d[2][j];
          BTd[1][j] = vaddq_s16(d[1][j], d[2][j]);

          // BTd[2][j] = d[2][j] - d[1][j];
          BTd[2][j] = vsubq_s16(d[2][j], d[1][j]);

          // BTd[3][j] = d[1][j] - d[3][j];
          BTd[3][j] = vsubq_s16(d[1][j], d[3][j]);
        }

        // U = BT d B
        for (int i = 0; i < tile_rows; i++)
        {
          // U[i][0] = BTd[i][0] - BTd[i][2];
          U[i][0] = vsubq_s16(BTd[i][0],BTd[i][2]);

          // U[i][1] = BTd[i][1] + BTd[i][2];
          U[i][1] = vaddq_s16(BTd[i][1], BTd[i][2]);

          // U[i][2] = BTd[i][2] - BTd[i][1];
          U[i][2] = vsubq_s16(BTd[i][2], BTd[i][1]);

          // U[i][3] = BTd[i][1] - BTd[i][3];
          U[i][3] = vsubq_s16(BTd[i][1], BTd[i][3]);
        }
        // Store the transformed matrix: HWC layout
        for (int i = 0, m = 0; i < tile_rows; i++)
        {
          for (int j = 0; j < tile_cols; j++, m++)
          {
            vst1q_s16(outptr + m*matrix_stride, U[i][j]); // matrix_stride is oc * num_tiles
          }
        }
        outptr += 8;
    }

    // if padding , memory overhead

    if (remaining_channels != 0){
        const size_t predecrement = 8 - remaining_channels;
        const int64x1_t v_shift = vmov_n_s64(-8 * predecrement);
        //
        // Winograd Input transform 4x4x4 tile
        // U = BT d B = (BT (BT d)T)T
        int16x8_t d[tile_rows][tile_cols];
        int16x8_t BTd[tile_rows][tile_cols];
        int16x8_t U[tile_rows][tile_cols];

        // load d
        for (int i = 0; i < tile_rows; i++)
        {
            for (int j = 0; j < tile_cols; j++)
            {
                const uint8x8_t vec = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(ptrs[i][j] - predecrement)), v_shift));
                d[i][j] = vreinterpretq_s16_u16(vsubl_u8(vec, va_zero_point));
            }
        }


        // BT d
        for (int j = 0; j < tile_cols; j++)
        {
          // BTd[0][j] = d[0][j] - d[2][j];
          BTd[0][j] = vsubq_s16(d[0][j], d[2][j]);

          // BTd[1][j] = d[1][j] + d[2][j];
          BTd[1][j] = vaddq_s16(d[1][j], d[2][j]);

          // BTd[2][j] = d[2][j] - d[1][j];
          BTd[2][j] = vsubq_s16(d[2][j], d[1][j]);

          // BTd[3][j] = d[1][j] - d[3][j];
          BTd[3][j] = vsubq_s16(d[1][j], d[3][j]);
        }

        // U = BT d B
        for (int i = 0; i < tile_rows; i++)
        {
          // U[i][0] = BTd[i][0] - BTd[i][2];
          U[i][0] = vsubq_s16(BTd[i][0],BTd[i][2]);

          // U[i][1] = BTd[i][1] + BTd[i][2];
          U[i][1] = vaddq_s16(BTd[i][1], BTd[i][2]);

          // U[i][2] = BTd[i][2] - BTd[i][1];
          U[i][2] = vsubq_s16(BTd[i][2], BTd[i][1]);

          // U[i][3] = BTd[i][1] - BTd[i][3];
          U[i][3] = vsubq_s16(BTd[i][1], BTd[i][3]);
        }
        // Store the transformed matrix: HWC layout
        if ( remaining_channels >= 4){
            for (int i = 0, m = 0; i < tile_rows; i++)
            {
              for (int j = 0; j < tile_cols; j++, m++)
              {
                vst1q_lane_s64(outptr + m*matrix_stride, vreinterpretq_s64_s16(U[i][j]), 0);
                U[i][j] = vextq_s16(U[i][j], U[i][j], 4);
              }
            }
            outptr += 4;
            remaining_channels -=4;
        }
        if ( remaining_channels >= 2 ){
            for (int i = 0, m = 0; i < tile_rows; i++)
            {
              for (int j = 0; j < tile_cols; j++, m++)
              {
                vst1q_lane_s32(outptr + m*matrix_stride, vreinterpretq_s32_s16(U[i][j]), 0);
                U[i][j] = vextq_s16(U[i][j], U[i][j], 2);
              }
            }
            outptr += 2;
            remaining_channels -= 2;
        }
        if ( remaining_channels != 0 ){
            for (int i = 0, m = 0; i < tile_rows; i++)
            {
              for (int j = 0; j < tile_cols; j++, m++)
              {
                vst1q_lane_s16(outptr + m*matrix_stride, U[i][j], 0);
              }
            }
        }
    }

}
