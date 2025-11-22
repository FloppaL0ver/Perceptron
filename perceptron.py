import matplotlib.pyplot as plt
import numpy as np

'''
    Making a sigmoid function. It's used to squash the range the input x (a number) to be between 0 to 1
    The formula is 1/[1+ exp(-x)] 

    As x approaches positive infinity, the sigmoid function returns a value close to 1 (exp(-x) tends to 0 as x becomes very big)
    As x approaches negative infinity, the sigmoid function returns a value close to 0 (exp(-x) tends to infinity as x becomes very negative)
    When x = 0 the sigmoid function returns 0.5 since exp(-x) turns to 1 and so 1/(1+1) = 0.5

    The sigmoid function is also differentiatable which is good since it helps us calculate gradient at certain points 
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#asks for learningRate
while True:
    try:
        learningRate = float(input("Enter the learningRate: "))
        #learningRate must be a postiive number (I limit it to <1 or else the graph gets spiky)
        if learningRate > 0 and learningRate < 1:
            #break exits the loop
            break
        else:
            print("Enter a positive number less than 1")
    except ValueError:
        print("ValueError; try again")

class perceptron:
    #__init__ is a function that automatically runs when something's class is set to this (perceptron)
    #I did x=learningRate so that in the later loop it wouldn't have to constantly key this value in
    def __init__(self, num_inputs, x=learningRate):
        '''weights should be between 1 and -1 to initialize (just to be safe). np.random is a module used to generate random numbers
        np.random.uniform(low, high, size) generates a certain amount (size) of random floats following a uniform distribution 
        meaning each number is as likely to show up as the next. It returns a NumPy array (which is better than a Python list since they can act as vectors)
        '''
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = 0
        self.learningRate = x

    '''function to feed the perceptron inputs from the train function below and use 
    its weights to calculate a sum (then squashing the output to be between 0 to 1 using the sigmoid function)  '''   
    def feed(self, inputs):
        '''
        This is actually very cool. So, numpy arrays can act as vectors (1D array). Inputs and self.weights are both numpy arrays.
        Therefore, if we wanted to sum them up we just find their dot product
        so if inputs was something like [a,b,c] and weights was [d,e,f] the dot product would be ad + be + cf 
        Then we just add the bias. The bias acts like a y-intercept (constant) so that the line wouldn't always pass through the origin
        '''
        sum = np.dot(inputs, self.weights) + self.bias
        #returns value from 0 to 1  and also the sum for extra info
        return sigmoid(sum), sum
    
    #function to train the perceptron, inputs into feed(); has an 'expected' value
    def train(self, input, expected):
        #feed returns 2 values so we only want the first output (the second value is just for debugging)
        output = (self.feed(input))[0]
        '''calculating the error by taking (output - expected) squared, getting mean-squared-error (MSE)
        the error is squared to remove negatives, and also to make bigger mistakes more punishing (it's also
        more convenient than using absolute since that has a 'V' shape and the gradient isn't smooth), this 
        expression is differentiable at every point which is important in calculating the cost gradient'''
        cost = (output - expected) ** 2
        #differentiating cost; 'grad' is the gradient, representing how much the cost changes given a change in weight dC/dW 
        #we find gradient because, to decrease the cost, we must go in the opposite direction to the gradient
        '''
        The gradient is probably the hardest mathematical concept in this project. It may look difficult. However, you only really need chain rule.
        We want to find the gradient for weight, which is dC/dW. So, we do dC/dW = dC/dO * dO/dx * dx/dW.
        Note that input is a numpy array so the corresponding weights will be affected by the corresponding gradients (numpy magic)
        '''
        #grad for weights
        wGrad = 2 * (output - expected) * output * (1-output) * input

        '''adjusting weights. If grad = 0 (i.e. the perceptron is fully correct) then no corrections would be made.
        If the perceptron overestimates the value then grad then it will adjust accordingly. grad * input[] to account
        for the sign of the input. 
        If you overestimate but then the input is negative then the weights will be increased since your grad is pos
        but your input is neg, so -= negative you're adding the magnitude of the number - this makes sense since increasing the weight
        would negative contribute to the total sum (since the input is negative). 
        If you overestimate but then the input is positive then the weights will be lowered since your grad will be pos
        learningRate is the (positive) factor by which the weights are corrected
        Because input was already converted to a numpy array, this means that we can just multiply it. What would happen is that
        every element in the numpy array would be multiplied with the same grad and self.LearningRate. The product of this series of operations
        would then be subtracted from the corresponding element in self.weights (another numpy array).
        Normally, in normal Python, I would have to do a for-loop in order to change the corresponding elements of an array. However,
        thanks to numpy magic, it just does it behind the scenes in optimised C code (or something like that) and it removes the hassle
        of having to write a long (and probably buggy) for-loop
        '''
        self.weights -= wGrad * input * self.learningRate

        #grad for bias (so that it's not a vector numpy array)
        '''
        The gradient for bias is slightly different. Instead of trying to find the change in cost w.r.t. change in weight, we find it w.r.t. change
        in bias. So dC/db. dC/db = dC/dO * dO/dx *dx/db
        '''
        bGrad = 2 * (output - expected) * output * (1-output)
        '''The bias is basically the y-intercept in output = (i1 * w1) + (i2 * w2) + bias where and i is the input and w is the weight
        We change the bias in the opposite direction to the gradient since that's how we minimise the cost
        e.g. if we overestimate the value then the gradient is positive (since gradient is 2 * (output - input) and our output > input)
        so we need to decrease the bias to nudge the output to the correct value (multiplied by learningrate to control how fast it changes)
        '''
        self.bias -= bGrad * self.learningRate
        return cost



#asks for number of inputs - for instance, if num_inputs = 2 then input may look something like [1 0 1] whereas if it's 3 input may be [1 0 0 1] (last digit is for expected value)
while True:
    try:
        num_inputs = int(input("Enter the number of inputs: "))
        break
    except ValueError:
        print("ValueError; try again")

#declaring that p is of the perceptron class
p = perceptron(num_inputs)


training_data = []
while True:
    try:
        #strip() removes any spaces before and after the start and end of the input
        #.split(",") splits the input into a list of strings (it will split them where a comma is present)
        raw_data = input(f"Enter values separated by spaces (last digit is the expected value between 0 - 1), each set separated by a comma: ").strip().split(",")

        for x in raw_data:
            #if it's blank
            if x.strip() == "":
                continue


            '''
            map(float, input) basically just makes everything in the list a float (decimal number)
            x.split() is used so that x, which could be something like (0.1 0.2 0.4 1) would be split to ["0.1", "0.2", "0.4", "1"]
            and then everything would be turned into floats, which works because no spaces are going to be attempted to convert to a float (which would happen if we didn't split)
            map() returns an 'iterator' which is something that you can only go through once
            So we use list() to turn it back into a list.
            np.array() (later on) converts this list into a numpy array which is useful for the dot product operation
            Even though numpy does accept python lists and converts them to numpy arrays internally, I thought doing this would be faster.
            inputs is a numpy array
            '''
            floats = list(map(float, x.split()))

            #[:-1] cuts out the last term (expected value) - giving us only the input values
            inputs = np.array(floats[:-1])

            #last number is the expected value as per stipulated by the input requirements
            expected = floats[-1]

            #if num_inputs is equal to 2 and the user puts something like [0 1 1 1] instead of [0 1 1] then it'll skip over it (remember that the last digit is the expected value)
            if len(inputs) != num_inputs:
                continue
        
            #expected value has to be between 0 and 1
            if not(expected <= 1 and expected >= 0):
                #if expected value does not fit then we don't use this set of data
                print("Please enter a value from 0-1")
                continue

            training_data.append((inputs, expected))

        if len(training_data) == 0:
            print("No valid training data provided.")
            exit()


        break
    except ValueError:
        print("ValueError; try again")




while True:
    try:
        epoch = int(input("Enter the number of training cycles: "))
        if epoch > 0:
            break
        else:
            print("Input a positive integer")
    except ValueError:
        print("ValueError; try again")

#running through each epoch (an epoch is like one cycle of training)
costs = []
#I initialise total_cost here so I could display average cost at the end. There's probably a better way to do this
total_cost = 0
for i in range(epoch):
    #total cost is reset to 0 each time so that we can get the total cost of each epoch (and so the average cost of each epoch) or else it would keep compounding
    total_cost = 0
    #trains perceptron and measures costs
    for inputs, expected in training_data:
        cost = p.train(inputs, expected)
        #we calculate the total cost so that we can find the average and graph it (or else the graph wouldn't really work)
        total_cost += cost
    costs.append(total_cost / len(training_data))

    #graphing in real-time
    #this changes the name of the window and makes a perceptron window
    plt.figure(num="Perceptron")
    #clf clears current figures so we get a new graph every time. Without this it would draw a line ontop of all the previous lines and they would overlap (messy)
    plt.clf()
    #plot.plot(x,y). In this case, the index is the x axis (epoch) and the values of the costs list is the y axis (cost)
    plt.plot(costs)
    plt.xlabel("Epoch")
    plt.ylabel("Average Cost")
    plt.title("Training")
    '''it pauses to redraw the graph; using pause is different from wait() since wait just literally pauses everything 
    but in pause it still processes GUI stuff without this you would only see the graph at the end'''
    plt.pause(0.00001)

#plt.show() basically leaves the graph open after training. During training, plt.pause() would redraw the graph in real time and after plt.show() keeps it open
plt.show()
print("Final weights:", p.weights)
print("Final bias:", p.bias)
print("Final average cost:", total_cost / len(training_data))