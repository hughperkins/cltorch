{{IndexType}} THClCeilDiv({{IndexType}} a, {{IndexType}} b) {
  return (a + b - 1) / b;
}
IndexType getStartIndex(IndexType totalSize) {
  IndexType sizePerBlock = THClCeilDiv(totalSize, (IndexType) gridDim.x);
  return blockIdx.x * sizePerBlock;
}

