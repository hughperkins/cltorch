"""
This does a first cut port from cutorch, into the `port` subdirectory.
I've never actually used it for porting whole files yet, but 
it does make using `meld` against newer cutorch branches, such as 
`goodies2` much more possible.
"""

from __future__ import print_function
import sys
import os

cutorch_dir = '../cutorch-goodies2'

cutorch_thc = '{cutorch_dir}/lib/THC'.format(
    cutorch_dir=cutorch_dir)

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
    f = open('port/lib/THCl/{filename}'.format(
        filename=filename), 'a')
    f.write('// from lib/THC/{filename}:\n\n'.format(
        filename=original_filename))
    contents = contents.replace('CUDA', 'CL')
    contents = contents.replace('Cuda', 'Cl')
    contents = contents.replace('#include "THC', '#include "THCl')
    contents = contents.replace('THC_', 'THCL_')
    contents = contents.replace('THCState', 'THClState')
    contents = contents.replace('CUTORCH', 'CLTORCH')
    contents = contents.replace('THCBlasState', 'THClBlasState')
    contents = contents.replace('cublasOperation_t', 'clblasTranspose')
    contents = contents.replace('cublas', 'clblas')
    contents = contents.replace('gridDim.x', '/*gridDim.x*/ get_num_groups(0)')
    contents = contents.replace('blockDim.x', '/*blockDim.x*/ get_local_size(0)')
    contents = contents.replace('blockIdx.x', '/*blockIdx.x*/ get_group_id(0)')
    contents = contents.replace('threadIdx.x', '/*threadIdx.x*/ get_local_id(0)')
    contents = contents.replace('__global__', 'kernel')
    contents = contents.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)')
    contents = contents.replace('warpSize', '{{WarpSize}}')
 
    # line by line:
    new_contents = ''
    scope_dead = False
    depth = 0
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
        new_contents += line + '\n'
    contents = new_contents

    f.write(contents)
    f.close()
    out_filenames.append(filename)


