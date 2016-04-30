kernel void THClStorageGet(global float *res, global float *data, int index) {
  if(get_global_id(0) == 0) {
    res[0] = data[index];
  }
}

