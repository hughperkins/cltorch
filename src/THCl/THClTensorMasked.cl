// from lib/THC/THCTensorMasked.cu:

struct TensorMaskedFillOp {
  TensorMaskedFillOp(float v) : value(v) {}
  /*__device__*/ /*__forceline__*/ void operator()(float* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      *t = value;
    }
  }

  float value;
};

struct TensorMaskedCopyOp {
  TensorMaskedCopyOp(float* s, float* bm, float* ps)
      : src(s),
        baseMask(bm),
        maskPrefixSum(ps) {
  }

  /*__device__*/ /*__forceline__*/ void operator()(float* out, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      // We've already checked that this offset is <= 2^24, so this is ok.
      int srcOffset = (int) (mask - baseMask);
      *out = src[(int) maskPrefixSum[srcOffset]];
    }
  }

  // Where we are copying from
  float* src;

  // The base address of mask so we can calculate offset
  float* baseMask;

  // The index we are copying from
  float* maskPrefixSum;
};

struct TensorMaskedSelectOp {
  TensorMaskedSelectOp(float* t) : out(t) {}
  /*__device__*/ /*__forceline__*/ void operator()(float* mask, float* maskPrefixSum, float* in) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      out[(int) *maskPrefixSum] = *in;
    }
  }

  float* out;
};

