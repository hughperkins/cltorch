kernel void THClStorageSet(global float *data, int index, float value) {
  if(get_global_id(0) == 0) {
//    int index2 = index;
//    data[index2] = 44;
    data[index] = value;
//    data[2] = index2;
//    data[3] = value;
  }
}

