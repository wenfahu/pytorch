#ifndef WINOGRAD
#define WINOGRAD

#include <arm_neon.h>
#include <qnnpack/params.h>
#include <qnnpack/math.h>

#ifdef __cplusplus
extern "C" {
#endif

void winograd_f_2_3_input_transform(
        const uint8_t* input_base,
        const int16_t* outptr,
        const int input_row_stride, // c * w
        const int input_col_stride, // c
        const int channels,
        const int matrix_stride, // num_tiles x c
        const uint8_t input_zero_point
		);

void winograd_f_2_3_kernel_transform(
        const uint8_t* input_base, // HWIO
        int16_t* output,
        const int input_channels,
        const int output_channels,
        const int matrix_stride,
        const int matrix_row_stride,
        const uint8_t kernel_zero_point
		);

void winograd_f_2_3_output_transform(
        const int32_t* input_base,
        const int32_t* bias_ptr,
        const uint8_t* output,
        const int n_channels,
        const int output_row_stride,
        const int output_col_stride,
        const int matrix_stride,
        const union pytorch_qnnp_conv_quantization_params* params
		);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
