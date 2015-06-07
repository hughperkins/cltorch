# clnn
Experimental opencl module for torch

## Things that are working

<table>

<tr><td>Component<td>Status<td>Examples</tr>

<tr><td><pre>require 'clnn'</pre> <td> works <td><pre>require 'clnn'</pre></tr>

<tr><td>Device information<td>works<td><pre>
print('num devices:', clnn.getDeviceCount())
props = clnn.getDeviceProperties(1)
</pre></tr>

<tr><td> torch.ClStorage <td> works <td><pre>
c = torch.ClStorage()
c = torch.ClStorage(3)
c[1] = 5
c = torch.ClStorage{4,9,2}
c:fill(7)
a = torch.Storage{1.5, 2.4, 5.3}
c:copy(a)
c[2] = 21
a:copy(c)
d = torch.ClStorage(3)
d:copy(c)
</pre></tr>

<tr><td>torch.ClTensor basic <td>works<td><pre>
c = torch.ClTensor{7,4,5}
c = torch.ClTensor(3,2)
c = torch.Tensor{2,6,9}:cl()
b = c:float()
c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
b:copy(c)
c:copy(b)
d = torch.ClTensor(2,3)
d:copy(c)
c[1][2] = 2.123
</pre></tr>

<tr><td>torch.ClTensor per-element maths <td>started<td><pre>
d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
c = torch.ClTensor{{4,2,-1},{3.1,1.2,4.9}}
c:add(d)
c:cmul(d)
c:cdiv(d)
c = c + d
c = c - d
c:abs()
</pre></tr>

</table>


