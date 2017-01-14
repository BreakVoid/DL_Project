require 'torch'
require 'image'
require 'nn'

print(sys.COLORS.red ..  '==> define parameters')

-- input dimensions
width = 120

print(sys.COLORS.red ..  '==> construct CNN')

local CNN = nn.Sequential()

-- stage 1:
CNN:add(nn.SpatialConvolutionMM(1, 20, 5, 1, 1, 1))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(2, 1, 2, 1))
CNN:add(nn.SpatialConvolutionMM(20, 50, 3, 1, 1, 1))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(2, 1, 2, 1))
CNN:add(nn.Reshape(50*28))

local classifier = nn.Sequential()
-- stage 2: Linear
classifier:add(nn.Linear(50*28, 500))
classifier:add(nn.Tanh())
classifier:add(nn.Linear(500, 4))
classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
  if layer.bias then
    layer.bias:fill(.2)
    if (i == #CNN.modules-1) then
      layers.bias:zero()
    end
  end
end

model = nn.Sequential()
model:add(CNN)
model:add(classifier)

loss = nn.ClassNLLCriterion()

print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

-- creating traindata
trainData={};
function trainData:size()
  return 400
end

file1 = io.open("./smooth_data/train_f", "r")
file2 = io.open("./original_data/train_l", "r")

for i = 1,trainData:size() do
  local input = torch.Tensor(1, 1, width)
  local output = torch.Tensor(1)
  for j = 1,width do
    input[1][1][j] = file1:read()
  end
  output[1] = file2:read() + 1
  trainData[i] = {input, output}
end
file1:close()
file2:close()


-- creating valdata
valData={}
function valData:size()
  return 40
end

file1 = io.open("./smooth_data/val_f", "r")
file2 = io.open("./original_data/val_l", "r")

for i = 1,valData:size() do
  local input = torch.Tensor(1, 1, width)
  local output = torch.Tensor(1)
  for j = 1,width do
    input[1][1][j] = file1:read()
  end
  output[1] = file2:read() + 1
  valData[i] = {input, output}
end
file1:close()
file2:close()

-- creating valdata
testData={}
function testData:size()
  return 228
end

file1 = io.open("./smooth_data/test_f", "r")
file2 = io.open("./original_data/test_l", "r")

for i = 1,testData:size() do
  local input = torch.Tensor(1, 1, width)
  local output = torch.Tensor(1)
  for j = 1,width do
    input[1][1][j] = file1:read()
  end
  output[1] = file2:read() + 1
  testData[i] = {input, output}
end
file1:close()
file2:close()
