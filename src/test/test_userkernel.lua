require 'cltorch'

local function eval(expression)
  loadstring('res=' .. expression)()
  print(expression, res)
end

eval('cltorch.kernel')
k = cltorch.kernel()
eval('k')

eval('torch.ClKernel')
src = [[
   bunch of source code...
]]
eval('src')
b = torch.ClKernel({src=src})
print('b', b)
eval('torch.ClKernel({src=' .. src .. '})')

