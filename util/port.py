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

cutorch_dir = '../cutorch-goodies2'

cutorch_thc = '{cutorch_dir}/lib/THC'.format(
    cutorch_dir=cutorch_dir)

def process_block(block):
    if block.find('__global__') >= 0 or block.find('__device__') >= 0:
        # kernel method, probably
        block = block.replace('gridDim.x', '/*gridDim.x*/ get_num_groups(0)')
        block = block.replace('blockDim.x', '/*blockDim.x*/ get_local_size(0)')
        block = block.replace('blockIdx.x', '/*blockIdx.x*/ get_group_id(0)')
        block = block.replace('threadIdx.x', '/*threadIdx.x*/ get_local_id(0)')
        block = block.replace('__global__', 'kernel')
        block = block.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)')
        block = block.replace('warpSize', '{{WarpSize}}')
        block = block.replace('IndexType', '{{IndexType}}')
        block = block.replace('__device__', '/*__device__*/')
        block = block.replace('__forceinline__', '/*__forceline__*/')
        return (block, True)
    return (block, False)

port_dir = 'port'
port_thc = '{port_dir}/lib/THCl'.format(
    port_dir=port_dir)
for filename in os.listdir(port_thc):
    os.remove('{port_thc}/{filename}'.format(
        port_thc=port_thc,
        filename=filename))
out_filenames = []
for filename in os.listdir(cutorch_thc):
    original_filename = filename
    print('filename', filename)
    f = open('{cutorch_thc}/{filename}'.format(
        cutorch_thc=cutorch_thc,
        filename=filename), 'r')
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
        f = open('port/lib/THCl/{filename}'.format(
            filename=filename), 'a')
        f.write('// from lib/THC/{filename}:\n\n'.format(
            filename=original_filename))
        f.write(new_contents)
        f.close()
        out_filenames.append(filename)
    if new_cl.strip() != '':
        clfilename = original_filename.replace('.cuh', '.cl')
        clfilename = clfilename.replace('.cu', '.cl')
        clfilename = clfilename.replace('THC', 'THCl')
        f = open('port/lib/THCl/{filename}'.format(
            filename=clfilename), 'a')
        f.write('// from lib/THC/{filename}:\n\n'.format(
        filename=original_filename))
        f.write(new_cl)
        f.close()

