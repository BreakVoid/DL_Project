require 'torch'
require 'image'
require 'nn'

print(sys.COLORS.red ..  '==> define parameters')

local noutputs = 4

-- input dimensions
local nfeats = 1
local width = 200

-- hidden units, filter sizes
local nstates = {32, 32}
local filtersize = {7, 7, 7}
local poolsize = 2

print(sys.COLORS.red ..  '==> construct CNN')


local CNN = nn.Sequential()

-- stage 1:
CNN:add(nn.TemporalConvolution(nfeats, nstates[1], filtersize[1]))
CNN:add(nn.Threshold())
CNN:add(nn.TemporalMaxPooling(poolsize, poolsize))

local classifier = nn.Sequential()

-- stage 2: Linear
classifier:add(nn.Linear(nstates[1], nstates[2]))
classifier:add(nn.Threshold())
classifier:add(nn.Linear(nstates[2], noutputs))

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


-- training function
function train(dataset)
  epoch = epoch or 1

  local time = sys.clock()

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,dataset:size(),opt.batchSize do
    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize,1,width)
    local targets = torch.Tensor(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end
  end
end

-- test function
function test(dataset)
  
end
