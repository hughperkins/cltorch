kernel void THClStorageSet(global float *data, long index, float value) {
  if(get_global_id(0) == 0) {
    data[index] = value;
  }
}

