require 'cltorch'

k = torch.ClKernel({input={nElements='int', input='torch.ClTensor'},output={output='torch.ClTensor'},src=[[
   int linearId = get_global_id(0);  // exciting stuff :-P
   if(linearId < nElements) {
     output_data[linearId] = input_data[linearId] + 3.0f;
   }
]]})
print('k', k)
k:print()

x = torch.ClTensor({3,5,2})
y = torch.ClTensor({6,4,2})
print('x before\n', x)
print('y before\n', y)

k:run({nElements=3, input=x, output=y})

print('y after\n', y)

