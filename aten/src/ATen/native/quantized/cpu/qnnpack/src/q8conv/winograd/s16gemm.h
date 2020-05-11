#ifndef S16GEMM
#define S16GEMM

#ifdef __cplusplus
extern "C" {
#endif

void s16gemm_kernel( size_t mr, // mr <= 8
        size_t nr, // nr <= 8
        size_t k, // kc
        const int16_t* a,
        size_t a_stride,
        const int16_t* b,
        int32_t* c,
        size_t c_stride);

void s16gemm(const int16_t*,
        const int16_t*, int32_t*, size_t, size_t, size_t);

#ifdef __cplusplus
} /* extern "C" */
#endif

static inline void pack_s16gemm_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  const int16_t* k,
  int16_t* packed_w)
{
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    packed_w += nr - nr_block_size;
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          *packed_w++ =
            k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
        }
        packed_w += kr - kr_block_size;
      }
      packed_w += (nr - nr_block_size) * kr;
    }
  }
}


#endif
