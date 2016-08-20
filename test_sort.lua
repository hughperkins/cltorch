require 'cltorch'

a = torch.FloatTensor(5):uniform()
--csorted = a:clone():sort(1)
gsorted = a:cl():sort():float()
print('gsorted', gsorted)
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)
----tester:asserteq(csorted, gsorted)

--a = torch.FloatTensor(55):uniform()
--csorted = a:clone():sort(1)
--gsorted = a:cl():sort():float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)
----tester:asserteq(csorted, gsorted)

--a = torch.FloatTensor(55):uniform()
--csorted = a:clone():sort(1)
--gsorted = a:cl():sort(1):float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)
----tester:asserteq(csorted, gsorted)





--  a = torch.FloatTensor(125):uniform()
--  csorted = a:clone():sort(1)
--  gsorted = a:cl():sort():float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)

--  a = torch.FloatTensor(125, 40):uniform()
--  csorted = a:clone():sort(1)
--  gsorted = a:cl():sort(1):float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)

--  a = torch.FloatTensor(5, 3):uniform()
--  csorted = a:clone():sort(2)
--  gsorted = a:cl():sort(2):float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)

--  a = torch.FloatTensor(50, 3):uniform()
--  csorted = a:clone():sort(2)
--  gsorted = a:cl():sort(2):float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)

--  a = torch.FloatTensor(125, 40):uniform()
--  csorted = a:clone():sort(2)
--  gsorted = a:cl():sort(2):float()
--diff = (csorted - gsorted):abs():max()
--assert(diff <= 1e-4)
----  print(csorted - gsorted)

