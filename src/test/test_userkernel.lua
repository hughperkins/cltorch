require 'cltorch'

local function eval(expression)
  loadstring('res=' .. expression)()
  print(expression, res)
end

eval('cltorch.kernel')
k = cltorch.kernel()
eval('k')

eval('torch.ClKernel')
b = torch.ClKernel({input={a='torch.ClTensor'},output={b='torch.ClTensor'},src=[[
   int linearId = get_global_id(0);  // exciting stuff :-P
]]})
print('b', b)
b:print()
-- eval('torch.ClKernel({src=' .. src .. '})')
b:run()

