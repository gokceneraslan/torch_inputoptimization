require 'torch'
require 'hdf5'
require 'nn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

--load pretrained network
model = torch.load('model.cpu')

--load the labels for which the input is going to be optimized
local l = hdf5.open('inputopt_labels.h5')
labels = l:read('labels_vec'):all()
model:float()

results = torch.FloatTensor(labels:size(2), 4, 1000, 1):zero()

state = {
   learningRate = 1e-2,
   momentum = 0.9,
   dampening = 0,
   nesterov = true,
   verbose = true
}



local truth = labels[{{}, 32}]:float()
local criterion = nn.BCECriterion()

--------------------------------
xx = torch.FloatTensor(4,1000,1):zero():float()
model:evaluate()
model:training()
model:zeroGradParameters()
local pred = model:forward(xx):float()
local err = criterion:forward(pred, truth)
local gradOut = criterion:backward(pred, truth)
model:updateGradInput(xx,gradOut)
---------------------------------

neval = 0
niter = 100000

local func = function(x)

    model:evaluate()
    pred = model:forward(x)
    err = criterion:forward(pred, truth)
    gradOut = criterion:backward(pred, truth)

    if neval % 100 == 0 then
        print('%' .. neval/(niter/100))
        print(err)
    end

    model:training()
    local g = model:updateGradInput(x, gradOut)
    neval = neval + 1
    bestx = x
    return err,g
end

bestx = torch.rand(4,1000,1)

for i=1,niter do

    optim.sgd(func,bestx,state)

end


--    results[i] = x:clone()
--
r = hdf5.open('results-optim-sgd.h5')
r:write('results', bestx:float())
r:close()
