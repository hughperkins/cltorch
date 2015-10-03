"""
This does a first cut port from `../cutorch-goodies2` directory,
into the `port` subdirectory.
I've never actually used it for porting whole files yet, but 
it does make using `meld` against newer cutorch branches, such as 
`goodies2` much more possible.

Possible future enhancements:
- make it automatically move kernels and device functions into a '.cl'
  file (plausibly anything with host goes both into the .h/.cpp, and also
  into the .cl)
"""

from __future__ import print_function
import sys
import os
from os.path import join as jp
from os import path

src_dir = '../cutorch'  # directory to port from

def process_block(block):
  if block.find(' operator()') >= 0:
    # its an Op struct, we are not writing these as kernels
    # but using Apply instead, and passing in the appropriate code
    # as strings into the kernel templates
    return (block, False)
  if block.find('__global__') >= 0 or block.find('__device__') >= 0:
    # kernel method, probably
    block = block.replace('gridDim.x', 'get_num_groups(0)')
    block = block.replace('gridDim.y', 'get_num_groups(1)')
    block = block.replace('blockDim.x', 'get_local_size(0)')
    block = block.replace('blockDim.y', 'get_local_size(1)')
    block = block.replace('blockIdx.x', 'get_group_id(0)')
    block = block.replace('blockIdx.y', 'get_group_id(1)')
    block = block.replace('threadIdx.x', 'get_local_id(0)')
    block = block.replace('threadIdx.y', 'get_local_id(1)')
    block = block.replace('__global__', 'kernel')
    block = block.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)')
    block = block.replace('warpSize', '{{WarpSize}}')
    block = block.replace('IndexType', '{{IndexType}}')
    block = block.replace('__device__', '/*__device__*/')
    block = block.replace('__forceinline__', '/*__forceline__*/')
    return (block, True)
  return (block, False)

def process_dir(cutorch_dir, port_dir, rel_dir):
  cutorch_src = jp(cutorch_dir, rel_dir)
  cltorch_dst = jp(port_dir, rel_dir).replace('THC', 'THCl')
  if not path.isdir(cltorch_dst):
    os.makedirs(cltorch_dst)
  for filename in os.listdir(cltorch_dst):
    filepath = jp(cltorch_dst, filename)
    if path.isfile(filepath):
      os.remove(filepath)
  out_filenames = []
  for filename in os.listdir(cutorch_src):
    original_filename = filename
    print('filename', filename)
    original_filepath = jp(cutorch_src, filename)
    if not path.isfile(original_filepath):
      continue
    f = open(jp(cutorch_src, filename), 'r')
    contents = f.read()
    f.close()
    base_name = filename.split('.')[0].replace('THC', 'THCl')
    suffix = '.' + filename.split('.')[1]
    if suffix == '.cuh':
      suffix = '.h'
    if suffix == '.cu':
      suffix = '.cpp'
    if suffix == '.c':
      suffix = '.cpp'
    filename = '{base}{suffix}'.format(
      base=base_name,
      suffix=suffix)
    if filename in out_filenames:
      print('warning: filename conflict: {filename}'.format(
        filename=filename))
    contents = contents.replace('CUDA', 'CL')
    contents = contents.replace('Cuda', 'Cl')
    contents = contents.replace('#include "THC', '#include "THCl')
    contents = contents.replace('THC_', 'THCL_')
    contents = contents.replace('THCState', 'THClState')
    contents = contents.replace('CUTORCH', 'CLTORCH')
    contents = contents.replace('THCBlasState', 'THClBlasState')
    contents = contents.replace('cublasOperation_t', 'clblasTranspose')
    contents = contents.replace('cublas', 'clblas')
    contents = contents.replace('cutorch', 'cltorch')
   
    # line by line:
    new_contents = ''
    new_cl = ''
    scope_dead = False
    depth = 0
    block = ''
    for line in contents.split('\n'):
      if line.startswith('#include <thrust'):
        line = '// ' + line
      elif line.find('thrust::') >= 0:
        line = '// ' + line
        scope_dead = True
      if line.find('{') >= 0:
        depth += 1
      if line.find('#include <cuda') >= 0:
        line = ''
      if line.strip() == 'THClCheck(cudaGetLastError());':
        line = ''
      if scope_dead and line.find('return') >= 0:
        line = ('  THError("Not implemented");\n' +
            '  return 0;\n  // ' +
            line)
        scope_dead = False
      if line.find('}') >= 0:
        if scope_dead:
          line = ('  THError("Not implemented");\n' +
            line)
          scope_dead = False
        depth -= 1
      block += line + '\n'
      if line.strip() == '' and depth == 0:
        block, is_cl = process_block(block)
        if is_cl:
          new_cl += block
        else:
          new_contents += block
        block = ''
    block, is_cl = process_block(block)
    if is_cl:
      new_cl += block
    else:
      new_contents += block
    block = ''
    if new_contents.strip() != "":
      f = open(jp(cltorch_dst, filename), 'a')
      f.write('// from lib/THC/{filename}:\n\n'.format(
        filename=original_filename))
      f.write(new_contents)
      f.close()
      out_filenames.append(filename)
    if new_cl.strip() != '':
      clfilename = original_filename.replace('.cuh', '.cl')
      clfilename = clfilename.replace('.cu', '.cl')
      clfilename = clfilename.replace('THC', 'THCl')
      clfilepath = jp(cltorch_dst, clfilename)
      f = open(clfilepath, 'a')
      f.write('// from {rel_dir}/{filename}:\n\n'.format(
        rel_dir=rel_dir,
        filename=original_filename))
      f.write(new_cl)
      f.close()

process_dir(src_dir, 'port', 'lib/THC')
process_dir(src_dir, 'port', 'torch')
process_dir(src_dir, 'port', 'torch/generic')
#  cutorch_dir = '../cutorch-goodies2'

#  cutorch_src = '{cutorch_dir}/lib/THC'.format(
#    cutorch_dir=cutorch_dir)

#  port_dir = 'port'

