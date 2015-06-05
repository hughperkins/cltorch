# clnn
Experimental opencl module for torch

This is 99.9999% experimental.  Might get deleted.  Who knows :-P

## Things that are working

<table>
<tr><td>Component<td>Status<td>Examples</tr>
<tr><td><pre>require 'clnn'</pre> <td> works <td><pre> `require 'clnn'` </pre></tr>
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
</table>


