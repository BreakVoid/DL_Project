require "nn"
require "model"
require "optim"
require "torch"
require "paths"
require "pl"
require "nnx"

local opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  -f,--full                                use the full dataset
  -p,--plot                                plot while training
  -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
  -r,--learningRate  (default 0.001)        learning rate, for SGD only
  -b,--batchSize     (default 20)          batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
  --coefL1           (default 0)
  L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  ]]

parameters,gradParameters = model:getParameters()
classes = {'1', '2', '3', '4'}
confusion = optim.ConfusionMatrix(classes)
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
criterion = loss

-- training function
function train(dataset)
  -- epoch init
  epoch = epoch or 1

  local time = sys.clock()
  shuffle = torch.randperm(dataset:size())

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,dataset:size(),opt.batchSize do
    -- create mini batch

    local inputs = torch.Tensor(opt.batchSize, 1, 1,width)
    local targets = torch.Tensor(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[shuffle[i]]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- just in case:
      collectgarbage()

      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- evaluate function for complete mini batch
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)

      -- estimate df/dW
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        -- locals:
        local norm,sign= torch.norm,torch.sign

        -- Loss:
        f = f + opt.coefL1 * norm(parameters,1)
        f = f + opt.coefL2 * norm(parameters,2)^2/2

        -- Gradients:
        gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
      end

      -- update confusion
      for i = 1,opt.batchSize do
        confusion:add(outputs[i], targets[i])
      end

      -- return f and df/dX
      return f,gradParameters
    end

    sgdState = sgdState or {
      learningRate = opt.learningRate,
      momentum = opt.momentum,
      learningRateDecay = 5e-7 }
    optim.sgd(feval, parameters, sgdState)

    -- disp progress
    xlua.progress(t, dataset:size())
  end


  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')


  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

  -- save/log current net
  local filename = paths.concat(opt.save, 'tone.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  if paths.filep(filename) then
    os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
  end
  print('<trainer> saving network to '..filename)
  -- torch.save(filename, model)

  -- next epoch
  epoch = epoch + 1
end


-- val function
function val(dataset)
  -- local vars
  local time = sys.clock()

  -- test over given dataset
  print('<trainer> on val Set:')

  for t = 1,dataset:size(),opt.batchSize do
    -- disp progress
    xlua.progress(t, dataset:size())

    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize,1,1,width)
    local targets = torch.Tensor(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end
    -- test samples
    local preds = model:forward(inputs)

    -- confusion:
    for i = 1,opt.batchSize do
      confusion:add(preds[i], targets[i])
    end
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  valLogger:add{['% mean class accuracy (val set)'] = confusion.totalValid * 100}
  confusion:zero()
end

-- test function
function test(dataset)
  -- local vars
  local time = sys.clock()

  -- test over given dataset
  print('<trainer> on testing Set:')

  for t = 1,dataset:size(),dataset:size() do
    -- disp progress
    xlua.progress(t, dataset:size())

    -- create mini batch
    local inputs = torch.Tensor(dataset:size(),1,1,width)
    local targets = torch.Tensor(dataset:size())
    local k = 1
    for i = t,math.min(t+dataset:size()-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end
    -- test samples
    local preds = model:forward(inputs)

    -- confusion:
    for i = 1,dataset:size() do
      confusion:add(preds[i], targets[i])
    end
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  valLogger:add{['% mean class accuracy (val set)'] = confusion.totalValid * 100}
  confusion:zero()
end

for i=1,50 do
  train(trainData)
  val(valData)
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
    valLogger:style{['% mean class accuracy (val set)'] = '-'}
    valLogger:plot()
  end
end



test(testData)
testLogger:style{['% mean class accuracy (test set)'] = '-'}
testLogger:plot()
