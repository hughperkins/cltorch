--print('done')

--require 'nn'

--a = nn.Linear(2,3)
--print(a)

--a:cl()

--b = torch.Storage(5)
--print(b)

-- c = torch.ClStorage(3)
-- print(c)

print("running require cltorch...")
require 'cltorch'
print("... require cltorch done")

props = cltorch.getDeviceProperties(1)
print('props', props)
for k,v in pairs(props) do
  print('props k,v', k, v)
end

print('num devices:', cltorch.getDeviceCount())

c = torch.ClStorage()
print('c1\n', c)

c = torch.ClStorage(3)
print('c2\n', c)
c[1] = 5
print('c3\n', c)
c[3] = 7
print('c4\n', c)
-- print('c' .. c)

c = torch.ClStorage{4,9,2}
print('c5\n', c)

c:fill(7)
print('c6\n', c)

a = torch.Storage{1.5, 2.4, 5.3}
print('a\n', a)

b = torch.Storage(3)
b:copy(a)
print('bbbbb\n', b)

c:copy(a)
print('c7\n', c)

c[2] = 21
print('c8\n', c)
print('aaaaaa\n', a)
a:copy(c)
print('aaaaaa\n', a)

d = torch.ClStorage(3)
d:copy(c)
print('dddd\n', d)

-- print(torch.Storage{3,5,2} == torch.Storage{3,5,2})


