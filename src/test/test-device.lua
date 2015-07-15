print("running require cltorch...")
require 'cltorch'
print("... require cltorch done")

numDevices =  cltorch.getDeviceCount()
print('num devices:', numDevices)

for device=1,numDevices do
  props = cltorch.getDeviceProperties(device)
  print('device properties, device', device)
  for k,v in pairs(props) do
    print('   ', k, v)
  end
end

for device=1,numDevices do
  cltorch.setDevice(device)
  c = torch.ClTensor{7,-4,5}
  print('c1\n', c)
  print(c:abs())
end

--c = torch.ClTensor{7,4,5}
--print('c1\n', c)



