# Sara Liu 4/27/19

import sys
import random
import math


def ff(weights, func, input):
    prevNodeCt = len(input)
    newInput = input[:]
    ffList = [input]
    for layer in range(len(weights)):
        layerWeights = [weight for weight in weights[layer]]
        if layer < len(weights) - 1:
            nodeWeights = [layerWeights[idx:idx + prevNodeCt] for idx in range(0, len(layerWeights), prevNodeCt)]
            nodeValues = []
            for node in nodeWeights:
                nodeValues.append(transferFunction(func, dotProduct(newInput, node)))
            newInput = nodeValues[:]
            prevNodeCt = len(newInput)
            # print('Layer ', layer + 1, ':')
            # print(*newInput)
            ffList.append(newInput)
        else:
            # print('Output:')
            output = []
            nodeCt = 0
            for weight in layerWeights:
                output.append(newInput[nodeCt] * weight)
                if nodeCt < len(newInput) - 1:
                    nodeCt += 1
            # print(*output)
            ffList.append(output)
    return ffList


def bp(ffList, weights, target):
    bpList = ffList[:]
    bpList[-1] = [target - ffList[-1][0]]
    for layer in range(len(bpList) - 2, 0, -1):
        bpList[layer] = [weights[layer][node] * bpList[layer + 1][0] * ffList[layer][node] * (1 - ffList[layer][node]) for node in range(len(ffList[layer]))]
    return bpList


def negGradient(ffList, bpList, numLayers):
    gradList = [[bpList[i + 1][idx // len(ffList[i])] * ffList[i][idx % len(ffList[i])] for idx in range(numLayers[i] * numLayers[i + 1])] for i in range(len(numLayers) - 1)]
    return gradList


def newWeights(weights, gradList):
    for weightLayer in range(len(weights)):
        for weight in range(len(weights[weightLayer])):
            weights[weightLayer][weight] = gradList[weightLayer][weight] * 0.01 + weights[weightLayer][weight]
    return weights


def dotProduct(vector1, vector2):
    if len(vector1) != len(vector2):
        return False
    dp = 0
    for idx in range(len(vector1)):
        dp += vector1[idx] * vector2[idx]
    return dp


def transferFunction(func, x):
    if func == 'T1':
        return x
    if func == 'T2':
        if x > 0:
            return x
        else:
            return 0
    if func == 'T3':
        return 1/(1 + math.exp(-x))
    if func == 'T4':
        return -1 + 2/(1 + math.exp(-x))
    else:
        return 'Function not valid'


file = open(sys.argv[1]).readlines()
inputs, targets = [[] for ipt in range(len(file))], []
for line in range(len(file)):
    splitLine = file[line].strip().split()
    pos = 0
    while splitLine[pos] != '=>':
        inputs[line].append(int(splitLine[pos]))
        pos += 1
    targets.append(int(splitLine[pos + 1]))
inputs = [inputList + [1] for inputList in inputs]
layerCts = [len(inputs[0]), 2, 1, 1]
weightList = [[random.uniform(-2.0, 2.0) for num in range(layerCts[i] * layerCts[i + 1])] for i in range(len(layerCts) - 1)]
errList = [10] * len(file)
errSum = sum(errList)
prevError = errSum
print('Layer cts: ', layerCts)
loopCt = 0
while True:
    testCase = loopCt % len(file)
    feedForward = ff(weightList, 'T3', inputs[testCase])
    # print(feedForward)
    finalErr = 0.5 * (targets[testCase] - feedForward[-1][0]) ** 2
    errList[testCase] = finalErr
    errSum = sum(errList)
    if errSum <= 0.01:
        break
    if loopCt % 1000 == 0:
        if abs(errSum - prevError) < 0.0001:
            weightList = [[random.uniform(-2.0, 2.0) for num in range(layerCts[i] * layerCts[i + 1])] for i in
                          range(len(layerCts) - 1)]
        else:
            prevError = errSum
    backPropagation = bp(feedForward, weightList, targets[testCase])
    # print(backPropagation)
    gradient = negGradient(feedForward, backPropagation, layerCts)
    # print(gradient)
    weightList = newWeights(weightList, gradient)
    loopCt += 1
print('Weights')
for wList in weightList:
    print(wList)
