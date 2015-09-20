kernel void THClStorageGet(global float *res, global float *data, long index) {
  if(get_global_id(0) == 0) {
    res[0] = data[index];
  }
}

