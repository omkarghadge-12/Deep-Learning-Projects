import numpy as np

def ReLU(Z):
    return max(0,Z)

def Neron_Forward(inputs , weights , bias):
    print("Inputs are (X): ",inputs)
    print("weights are (W): ",weights)
    print("bias are (b): ",bias)

    z = 0
    for i in range(len(inputs)):
        z = z + (inputs[i] * weights[i])

    z = z + bias

    #z = sum(w * x for w , x in zip(weights , inputs)) + bias
    print("Weighted sum : ",z)

    y =  ReLU(z)

    return y

def main():
    print("-----Neural Network-----")
    inputs = [1.0 , 2.0 , 3.0]
    weights = [0.6 , 0.4 , -0.2]
    bias = 0.5

    result = Neron_Forward(inputs , weights , bias)

    print("Predicted result : ",result)

if __name__ == "__main__":
    main()