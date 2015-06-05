# clnn
Experimental opencl module for torch

This is 99.9999% experimental.  Might get deleted.  Who knows :-P

## Things that are working

|  Component | Status | Examples |
|--|---|--|
| require 'clnn' | works | `require 'clnn' |
| device information | works | ```
print('num devices:', clnn.getDeviceCount())
props = clnn.getDeviceProperties(1)
print('props', props)
for k,v in pairs(props) do
  print('props k,v', k, v)
end
``` |
| torch.ClStorage | works | ```
c = torch.ClStorage()
print('c1\n', c)
c = torch.ClStorage(3)
print('c2\n', c)
c[1] = 5
print('c3\n', c)
c[3] = 7
print('c4\n', c)
c = torch.ClStorage{4,9,2}
print('c5\n', c)
c:fill(7)
print('c6\n', c)
a = torch.Storage{1.5, 2.4, 5.3}
print('a\n', a)
c:copy(a)
print('c7\n', c)
c[2] = 21
print('c8\n', c)
a:copy(c)
print('a\n', a)
d = torch.ClStorage(3)
d:copy(c)
print('d\n', d)
```

