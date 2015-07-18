require 'cltorch'

local function eval(expression)
  loadstring('res=' .. expression)()
  print(expression, res)
end

eval('cltorch.kernel')
k = cltorch.kernel()
eval('k')

t = {input={a='torch.ClTensor'}}
for k,v in pairs(t.input) do
  print('k,v', k, v)
end

eval('torch.ClKernel')
b = torch.ClKernel({input={nElements='int', input='torch.ClTensor'},output={output='torch.ClTensor'},src=[[
   int linearId = get_global_id(0);  // exciting stuff :-P
   if(linearId < nElements) {
     output_data[linearId] = input_data[linearId] = 3.0f;
   }
]]})
print('b', b)
b:print()
-- eval('torch.ClKernel({src=' .. src .. '})')

x = torch.ClTensor({3,5,2})
y = torch.ClTensor({6,4,2})
print('x before\n', x)
print('y before\n', y)

b:run({nElements=3, input=x, output=y})

print('y after\n', y)

