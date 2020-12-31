# Sara Liu 4/13/19

import sys
import math


def nn1(weights, func, input):
    prevNodeCt = len(input)
    newInput = input[:]
    for layer in range(len(weights)):
        layerWeights = [float(weight) for weight in weights[layer].strip().split()]
        if layer < len(weights) - 1:
            nodeWeights = [layerWeights[idx:idx + prevNodeCt] for idx in range(0, len(layerWeights), prevNodeCt)]
            nodeValues = []
            for node in nodeWeights:
                nodeValues.append(transferFunction(func, dotProduct(newInput, node)))
            newInput = nodeValues[:]
            prevNodeCt = len(newInput)
            print('Layer ', layer + 1, ':')
            print(*newInput)
        else:
            print('Output:')
            output = []
            nodeCt = 0
            for weight in layerWeights:
                output.append(newInput[nodeCt] * weight)
                if nodeCt < len(newInput) - 1:
                    nodeCt += 1
            print(*output)


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


file, transferFunc, inputs = [], '', []
for arg in range(len(sys.argv)):
    if arg == 1:
        file = open(sys.argv[arg]).readlines()
    elif arg == 2:
        transferFunc = sys.argv[arg]
    elif arg >= 3:
        inputs.append(float(sys.argv[arg]))
nn1(file, transferFunc, inputs)
