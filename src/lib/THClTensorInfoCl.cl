typedef struct THClTensorInfoCl {
  unsigned int sizes[{{MAX_CLTORCH_DIMS}}];
  unsigned int strides[{{MAX_CLTORCH_DIMS}}];
  int offset;
  int dims;
} TensorInfoCl;

