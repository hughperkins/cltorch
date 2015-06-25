-- I reckon that generating on host side and copying to gpu
-- will work just fine for many scenarios, eg dropout
-- I cite as evidence the answer by Klerik at
-- http://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl

function torch.ClTensor:bernoulli(p)
  if p ~= nil then
    self:copy(torch.Tensor(self:size()):bernoulli(p))
  else
    self:copy(torch.Tensor(self:size()):bernoulli())
  end
  return self
end

function torch.ClTensor:uniform(a, b)
  if a == nil then
    a = 0
  end
  if b == nil then
    b = 1
  end
  self:copy(torch.Tensor(self:size()):uniform(a, b))
  return self
end

