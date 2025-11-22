# Simple Perceptron
#### Video Demo:  <URL HERE>
#### Description:
This is a simple perceptron which accepts inputs (and expected values) and applies weights to them. Then, it sums up the weights and adds bias.  
Then, it's run through a sigmoid function to squash the value of the sum to be between 0 and 1. We want the output to be squashed so that it can be compared to our expected value (which is a value from 0 to 1).

The purpose of this project was to explore machine learning at a simple level that was achievable with no machine learning libraries needed.

The sigmoid function is simply:

$$
O = \frac{1}{1 + e^{-x}}
$$

Where
$$
x = W \cdot i + b
$$


The cost (or mean-squared error) is calculated through:

$$
C = (O - Y)^2
$$

The gradient for cost w.r.t. weight is:

$$
\frac{dC}{dW} = 2 \cdot (O - Y) \cdot O \cdot (1 - O) \cdot i
$$

Because cost is a function of the weight, the gradient is $dC/dW$. The derivation of this is quite lengthy, but essentially you're finding $dC/dW$ from the chain rule:

$$
\frac{dC}{dW} = \frac{dC}{dO} \cdot \frac{dO}{dX} \cdot \frac{dX}{dW}
$$

We have to calculate different gradients w.r.t. cost and bias and can't use the same one (1st because $dC/dW$ is different from $dC/db$ and 2nd because input is a numpy array so when we multiply input, the result acts like a vector and would create multiple lines in the graph which we don't want happening).  

![weight gradient](weightGrad.png)  

The gradient for cost w.r.t. bias is:
$$
\frac{dC}{dW} = 2 \cdot (O - Y) \cdot O \cdot (1 - O)
$$

![bias gradient](biasGrad.png)

We find the gradient so that we can go in the opposite direction to reduce the cost.  
If the perceptron overestimates the output, then the weights and bias will be adjusted accordingly:

$$
\text{self.weights} -= \text{wGrad} \cdot \text{input} \cdot \text{learningRate}
$$

$$
\text{self.bias} -= \text{bGrad} \cdot \text{learningRate}
$$

This is what enables the perceptron to "learn".  
It basically just changes its weights and bias based on that equation until its cost is minimized.  
The equations work for multiple weights because of the magic of numpy arrays (they just do subtraction and addition as if the arrays were vectors).

---

A graph is drawn using matplotlib, showing average cost and epoch (training cycles).  
Every cycle, the perceptron is trained and then its average cost:

$$
\text{Average Cost} = \frac{\text{Total Cost}}{\text{Number of Training Samples}}
$$

is calculated.

Essentially, the perceptron is finding 'best-fit' values for all of the training data provided; after many cycles, it will eventually find a combination which minimises the error for all of the data the best.

Because this is single-layer perceptron, it has many limitations.
For instance, if the output depends on a specific combination of inputs to occur, then this perceptron cannot represent that:
Here, the expected value is 1 if exactly two inputs are 1. A perceptron cannot 'learn' from this well (copy and paste AS ONE LINE; 4 inputs).
```
0 0 0 0 0,
0 0 0 1 0,
0 0 1 0 0,
0 0 1 1 1,
0 1 0 0 0,
0 1 0 1 1,
0 1 1 0 1,
0 1 1 1 0,
1 0 0 0 0,
1 0 0 1 1,
1 0 1 0 1,
1 0 1 1 0,
1 1 0 0 1,
1 1 0 1 0,
1 1 1 0 0,
1 1 1 1 0
```

Additionally, data which is non-linear would not be able to be used on this perceptron 

(2 inputs):
```
0.0 0.0 0, 0.1 0.0 0.309, 
0.2 0.0 0.588, 0.3 0.0 0.809, 
0.4 0.0 0.951, 0.5 0.0 1, 
0.6 0.0 0.951, 0.7 0.0 0.809, 
0.8 0.0 0.588, 0.9 0.0 0.309, 
1.0 0.0 0
```

Only data that can be separated linearly, and are consistent (meaning there are no conflicting inputs) can be effectively used for this simple perceptron. 

Sample inputs (just copy and paste AS ONE LINE); I asked ChatGPT to provide me with interesting examples:
2 input:
```
0 0 0, 0 1 1, 1 0 1, 
1 1 1, 0.2 0.5 0.6, 0.7 0.1 0.8, 
0.4 0.9 0.9, 0.8 0.8 1
```
input that may not work well:
```
0 0 0, 0 1 1, 1 0 1, 1 1 0
0 0 0, 0 1 0, 1 0 0, 1 1 1
0 0 0, 0 0 1
```



3 input:
```
0 0 0 0, 0 0 1 0, 0 1 0 0, 
1 0 0 0, 0 1 1 1, 1 0 1 1, 
1 1 0 1, 1 1 1 1
```
inputs that may not work well:
```
0 0 0 0, 0 0 1 0, 0 1 0 0, 
1 0 0 0, 0 1 1 1, 1 0 1 1, 
1 1 0 1, 1 1 1 0
```


4 input:
```
0 0 0 0 0, 0 0 0 1 0, 0 0 1 0 0, 
0 1 0 0 0, 1 0 0 0 0, 0 0 1 1 0, 
0 1 0 1 0, 1 0 0 1 0, 0 1 1 0 0, 
1 0 1 0 0, 1 1 0 0 0, 0 1 1 1 1, 
1 0 1 1 1, 1 1 0 1 1, 1 1 1 0 1, 
1 1 1 1 1
```

```
0.1 0.2 0.3 0.4 0.25, 0.5 0.3 0.7 0.2 0.55, 
0.6 0.1 0.4 0.9 0.65, 0.8 0.7 0.2 0.5 0.75,
0.9 0.3 0.6 0.8 0.85, 0.2 0.5 0.1 0.9 0.45, 
0.3 0.7 0.8 0.2 0.65, 0.4 0.6 0.9 0.1 0.7
```

inputs that may not work well:
```
0 0 1 1 1, 0 1 1 0 1, 1 1 0 0 1, 1 0 1 0 1, 
0 0 0 1 0, 1 1 1 0 0, 0 1 0 1 0, 1 0 0 0 0
```

