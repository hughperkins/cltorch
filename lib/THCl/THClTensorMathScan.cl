// from lib/THC/THCTensorMathScan.cu:

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same get_group_id(1) process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */

inline float binary_op(float _in1, float _in2) {
  // hope the compiler can handle this :-P
  float _out;
  float *out = &_out;
  float *in1 = &_in1;
  float *in2 = &_in2;
  {{operator3}};
  return _out;
}

kernel void THClTensor_kernel_scanOuterDim(
  global float *tgt_data, int tgt_offset,
  global float *src_data, int src_offset,
  int num_orows, int num_irows, int row_size,
  float init)
{
  for (unsigned orow = get_group_id(0); orow < num_orows; orow += get_num_groups(0)) {
    for (unsigned irow = get_group_id(1) * get_local_size(0) + get_local_id(0); irow < num_irows; irow += get_num_groups(1) * get_local_size(0)) {
      global float *src = src_data + src_offset + orow * row_size * num_irows + irow;
      global float *tgt = tgt_data + tgt_offset + orow * row_size * num_irows + irow;
      float acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        acc = binary_op(acc, *src);
//        binary_op(&acc, &acc, src);
        *tgt = acc;

        src += num_irows;
        tgt += num_irows;
      }
    }
  }
}

/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
//template<int num_threads_x, int num_threads_y, class BinaryFunction>
kernel void THClTensor_kernel_scanInnermostDim(
  global float *tgt_data, int tgt_offset,
  global float *src_data, int src_offset,
  int num_rows, int row_size,
  float init)
{
  local float sbuf[{{num_threads_y}}][2 * {{num_threads_x}}];

  local float* row_buf = sbuf[get_local_id(1)];

  for (int block_row = get_group_id(0) * get_local_size(1);
       block_row < num_rows;
       block_row += get_local_size(1) * get_num_groups(0)) {
    int row = block_row + get_local_id(1);
    float block_total = init;

    global float *row_src = src_data + src_offset + row * row_size;
    global float *row_tgt = tgt_data + tgt_offset + row * row_size;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (int block_col = 0; block_col < row_size; block_col += 2 * {{num_threads_x}}) {
      // Load data into shared memory (two values per thread).
      int col1 = block_col + get_local_id(0);
      int col2 = block_col + {{num_threads_x}} + get_local_id(0);
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[get_local_id(0)] = row_src[col1];
        } else {
          row_buf[get_local_id(0)] = init;
        }

        if (col2 < row_size) {
          row_buf[{{num_threads_x}} + get_local_id(0)] = row_src[col2];
        } else {
          row_buf[{{num_threads_x}} + get_local_id(0)] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (get_local_id(0) == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
//          binary_op(row_buf, row_buf, &block_total);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Parallel reduction (up-sweep).
      for (int s = {{num_threads_x}}, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && get_local_id(0) < s) {
          int offset = (2 * get_local_id(0) + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
//          binary_op(row_bufer + offset + d, row_buf + offset, row_buf + offset + d);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      // Down-sweep.
      for (int s = 2, d = {{num_threads_x}} / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && get_local_id(0) < s - 1) {
          int offset = 2 * (get_local_id(0) + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
//          binary_op(row_buff + offset + d, row_buf + offset, row_buf + offset + d);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }


      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[get_local_id(0)];
        if (col2 < row_size) row_tgt[col2] = row_buf[{{num_threads_x}} + get_local_id(0)];
      }
      block_total = row_buf[2 * {{num_threads_x}} - 1];
      barrier(CLK_LOCAL_MEM_FENCE);

    }
  }
}

