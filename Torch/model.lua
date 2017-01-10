require 'torch'
require 'image'
require 'nn'

print(sys.COLORS.red ..  '==> define parameters')

local noutputs = 4

-- input dimensions
nfeats = 1
width = 200

-- hidden units, filter sizes
local nstates = {1, 32}
local filtersize = {7, 7, 7}
local poolsize = 2

print(sys.COLORS.red ..  '==> construct CNN')

local CNN = nn.Sequential()

-- stage 1:
--[[
CNN:add(nn.TemporalConvolution(nfeats, 1, filtersize[1]))
CNN:add(nn.Threshold())
CNN:add(nn.TemporalMaxPooling(poolsize))
]]--

local classifier = nn.Sequential()
-- stage 2: Linear
--[[
classifier:add(nn.Linear(width, nstates[2]))
classifier:add(nn.Threshold())
classifier:add(nn.Linear(nstates[2], noutputs))
]]--
classifier:add(nn.Linear(width, noutputs))
classifier:add(nn.LogSoftMax())
--[[
for _,layer in ipairs(CNN.modules) do
  if layer.bias then
    layer.bias:fill(.2)
    if (i == #CNN.modules-1) then
      layers.bias:zero()
    end
  end
end
]]--
model = nn.Sequential()
--model:add(CNN)
model:add(classifier)

loss = nn.ClassNLLCriterion()

print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

-- creating traindata
trainData={};
function trainData:size()
  return 400
end

file1 = io.open("train_f0s", "r")
file2 = io.open("train_labels", "r")

for i = 1,trainData:size() do
  local input = torch.Tensor(width)
  local output = torch.Tensor(1)
  for j = 1,width do
    input[j] = file1:read()
  end
  output[1] = file2:read() + 1
  trainData[i] = {input, output}
end
file1:close()
file2:close()


-- creating testdata
testData={}
function testData:size()
  return 400
end

file1 = io.open("val_f0s", "r")
file2 = io.open("val_labels", "r")

for i = 1,testData:size() do
  local input = torch.Tensor(width)
  local output = torch.Tensor(1)
  for j = 1,width do
    input[j] = file1:read()
  end
  output[1] = file2:read() + 1
  testData[i] = {input, output}
end
file1:close()
file2:close()

