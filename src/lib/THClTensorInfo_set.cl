{{include_THClReduceApplyUtils}}

#define TSIZEINTS (sizeof(TensorInfoCl) / sizeof(int))
kernel void set(int index, global int *infos, global int *newInfo) {
  if(get_global_id(0) >= TSIZEINTS) {
    return;
  }
  infos[index * TSIZE + get_global_id(0)] = newInfo[get_global_id(0)];  // copy it over as ints?
}

