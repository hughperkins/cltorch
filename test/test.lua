--print('done')

--require 'nn'

--a = nn.Linear(2,3)
--print(a)

--a:cl()

--b = torch.Storage(5)
--print(b)

-- c = torch.ClStorage(3)
-- print(c)

-- this function doesnt work
function table_to_str(table)
  results = '{'
  for k,v in pairs(table) do
    results = results .. k .. '=' .. v .. ' '
    -- print(k, ':', v)
  end
  results = results .. '}'
  return results
end

print("running require clnn...")
require 'clnn'
print("... require clnn done")

for k,v in pairs(clnn) do
  print('clnn k,v', k, v)
end

props = clnn.getDeviceProperties(1)
print('props', props)
for k,v in pairs(props) do
  print('props k,v', k, v)
end

