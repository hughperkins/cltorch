// from lib/THC/THCTensorMath2.cu:

// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<bool flag, bool apply_sqrt>
/*__forceline__*/ /*__device__*/ float THClTensor_computeVar(float sum, float sum2, unsigned row_size) {
  if (flag) {
    sum /= row_size;
    sum2 /= row_size;
    sum2 -= sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  else {
    sum /= row_size;
    sum2 /= row_size - 1;
    sum2 -= ((float)row_size) / ((float)(row_size - 1)) * sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  if (apply_sqrt)
    return sqrt(sum2);
  else
    return sum2;
}

/* Compute the variance (or standard deviation) along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same get_group_id(1) process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<bool flag, bool apply_sqrt>
kernel void THClTensor_kernel_varOuterDim(float *tgt, float *src_, unsigned num_orows, unsigned num_irows, unsigned row_size)
{
  for (unsigned orow = get_group_id(0); orow < num_orows; orow += get_num_groups(0)) {
    for (unsigned irow = get_group_id(1) * get_local_size(0) + get_local_id(0); irow < num_irows; irow += get_num_groups(1) * get_local_size(0)) {
      float *src = src_ + orow * row_size * num_irows + irow;
      float sum = 0, sum2 = 0;

      for (unsigned col = 0; col < row_size; ++col) {
        float val = *src;
        sum += val;
        sum2 += val * val;

        src += num_irows;
      }

      tgt[orow * num_irows + irow] = THClTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
  }
}

/* Compute the variance (or standard deviation) of the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<bool flag, bool apply_sqrt>
kernel void THClTensor_kernel_varInnermostDim(float *tgt, float *src_, unsigned num_rows, unsigned row_size)
{
  local float ssum[32][16];
  local float ssum2[32][16];

  for (unsigned block_row = get_group_id(0) * get_local_size(1); block_row < num_rows; block_row += get_local_size(1) * get_num_groups(0)) {
    unsigned row = block_row + get_local_id(1);
    float sum = 0, sum2 = 0;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = get_local_id(0); col < row_size; col += get_local_size(0)) {
        float val = src[col];
        sum += val;
        sum2 += val * val;
      }
    }
    ssum[get_local_id(1)][get_local_id(0)] = sum;
    ssum2[get_local_id(1)][get_local_id(0)] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce intermediate values to single value.
    for (unsigned s = 8; s > 1; s >>= 1) {
      if (row < num_rows && get_local_id(0) < s) {
        ssum[get_local_id(1)][get_local_id(0)] += ssum[get_local_id(1)][get_local_id(0) + s];
        ssum2[get_local_id(1)][get_local_id(0)] += ssum2[get_local_id(1)][get_local_id(0) + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < num_rows && get_local_id(0) == 0) {
      sum = ssum[get_local_id(1)][0] + ssum[get_local_id(1)][1];
      sum2 = ssum2[get_local_id(1)][0] + ssum2[get_local_id(1)][1];
      tgt[row] = THClTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

kernel void THClTensor_kernel_renorm(float *data, const float value, const long size, const float maxnorm)
{
  local float buffer[32];
  long tx = get_local_id(0);
  long bx = get_group_id(0);
  long step = get_local_size(0);
  float *row = data + size*bx;

  buffer[tx] = 0;

  // get norm of axis
  for (long i=tx; i<size; i+=step)
  {
    buffer[tx] += pow(fabs(row[i]), value);
  }
  // add (reduce)
  for (unsigned int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  // clip norms
  barrier(CLK_LOCAL_MEM_FENCE);
  float norm = pow(buffer[0], 1/value);
  if (norm > maxnorm)
  {
    norm = maxnorm / (norm + 1e-7);
    // renormalize
    for (long i=tx; i<size; i+=step)
    {
      row[i] *= norm;
    }
  }
}

