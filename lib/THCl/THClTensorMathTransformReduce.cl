// from lib/THC/THCTensorMathTransformReduce.cu:

typedef struct Pair {
   float first;
   float second;
} Pair;

Pair binary_op( Pair a, Pair b ) {
    {{pair_operator2}};
}

/* A set of reduction kernels that take in binary ops on thrust pairs (of value, index).
   These are useful when you not only have to do a reduction, but you might have
   to preserve the location of contention (for example min/max operations).
   The structure of the kernels follows the structure of the reduction kernels.
*/
kernel void THClTensor_kernel_transformReduceOuterDimIndex(global float *tgt1_data,
                                                          int tgt1_offset,
                                                          global float *tgt2_data,
                                                          int tgt2_offset,
                                                             global float *src_data,
                                                           int src_offset,
                                                             int num_orows,
                                                             int num_irows,
                                                             int row_size)
{
  for (int orow = get_group_id(0); orow < num_orows; orow += get_num_groups(0)) {
    for (int irow = get_group_id(1) * get_local_size(0) + get_local_id(0); irow < num_irows; irow += get_num_groups(1) * get_local_size(0)) {
      global float *src = src_data + src_offset + orow * row_size * num_irows + irow;
      Pair acc = {.first={{init}}, .second=-1};
      for (int col = 0; col < row_size; ++col) {
        Pair lhs = {*src, col+1};
        acc = binary_op( lhs, acc);
//         acc = binary_op(thrust::make_pair(*src, col+1), acc); // i+1 for 1-indexing
        src += num_irows;
      }
      tgt1_data[tgt1_offset + orow * num_irows + irow] = acc.first;
      tgt2_data[tgt2_offset + orow * num_irows + irow] = acc.second;
    }
  }
}

/* Reduce the innermost dimension of a tensor (on thrust::pair functors which are (value, index))
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
kernel void THClTensor_kernel_transformReduceInnermostDimIndex(
  global float *tgt1_data, int tgt1_offset, global float* tgt2_data, int tgt2_offset,
  global float *src_data, int src_offset,
  int num_rows, int row_size )
{
  local float sbuf[{{y_threads}}][{{x_threads}}];
  local float ibuf[{{y_threads}}][{{x_threads}}];

  for (int block_row = get_group_id(0) * get_local_size(1); block_row < num_rows; block_row += get_local_size(1) * get_num_groups(0)) {
    int row = block_row + get_local_id(1);
//     thrust::pair<float,float> acc = init;
    Pair acc = { .first={{init}}, .second=-1 };
    if (row < num_rows) {
      global float *src = src_data + src_offset + row * row_size;
      // Sequential reduction within a thread.
      for (int col = get_local_id(0); col < row_size; col += get_local_size(0)) {
        Pair lhs = {src[col], col+1};
        acc = binary_op(lhs, acc);
      }
    }

    sbuf[get_local_id(1)][get_local_id(0)] = acc.first;
    ibuf[get_local_id(1)][get_local_id(0)] = acc.second;

    // Reduce intermediate values to single value.
    local float* sline = &sbuf[get_local_id(1)][0];
    local float* iline = &ibuf[get_local_id(1)][0];
    for (int s = 8; s > 0; s >>= 1) {
      if (row < num_rows && get_local_id(0) < s) {
        Pair arg1 = {.first=sline[get_local_id(0)], .second=iline[get_local_id(0)]};
        Pair arg2 = {.first=sline[get_local_id(0) + s], .second=iline[get_local_id(0) + s]};
        Pair res = binary_op(arg1, arg2);
        sline[get_local_id(0)] = res.first;
        iline[get_local_id(0)] = res.second;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < num_rows && get_local_id(0) == 0) {
      tgt1_data[row + tgt1_offset] = sline[0];
      tgt2_data[row + tgt2_offset] = iline[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

