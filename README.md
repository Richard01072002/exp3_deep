# EXP-3
### Name- RICHARDSON A
### reg no. - 212222233005
### dep - AI & DS 
## AIM
#### To train a linear regression model using PyTorch to learn the relationship y = 2X + 1 + noise and optimize it using MSE loss and SGD, then evaluate its performance by plotting the best-fit line.

#### PROCEDURE
#### Step 1: Data Preparation: A dataset was generated using X = torch.linspace(1,50,50).reshape(-1,1), with random noise e added to create the target variable y = 2X + 1 + e. The data was visualized using a scatter plot.

#### Step 2: Model Definition: A simple linear regression model was defined using nn.Linear() with one input and one output feature. The model's weight and bias were initialized randomly.

#### Step 3: Training Process: The model was trained using Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimizer with a learning rate of 0.001. Training was performed over 50 epochs, with weight and bias values updated iteratively. The loss decreased steadily, showing that the model was learning.

#### Step 4: Evaluation and Visualization: After training, the model parameters were extracted. The final weight was approximately 2.0, and the bias was approximately 1.0, closely matching the expected values. A best-fit line was plotted over the original data, demonstrating the model’s ability to generalize the relationship y = 2X + 1 effectively.

## PROGRAM
```
iimport torch
import torch.nn as nn  # Neural network module

import numpy as np
import matplotlib.pyplot as plt  # For plotting
%matplotlib inline  
```


```
X = torch.linspace(1,50,50).reshape(-1,1)
```


```
torch.manual_seed(71) # to obtain reproducible results
e = torch.randint(-8,9,(50,1),dtype=torch.float)
```



```
y=2*X+1+e
print(y.shape)
```
#### output 1
<img width="241" alt="Screenshot 2025-03-06 at 2 47 07 PM" src="https://github.com/user-attachments/assets/46509fd2-abad-46e5-85c7-9757fd1cea9c" />



```
plt.scatter(X.numpy(), y.numpy(),color='red')  # Scatter plot of data points
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
```
#### output 2
<img width="575" alt="Screenshot 2025-03-06 at 2 47 55 PM" src="https://github.com/user-attachments/assets/260e0b30-61d1-450a-b210-aeb2b3311c7c" />


```
# Setting a manual seed for reproducibility
torch.manual_seed(59)

# Defining the model class
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
```

```
# Creating an instance of the model
torch.manual_seed(59)
model = Model(1, 1)
print('Weight:', model.linear.weight.item())
print('Bias:  ', model.linear.bias.item())
```
#### output 3
<img width="292" alt="Screenshot 2025-03-06 at 2 48 21 PM" src="https://github.com/user-attachments/assets/791d9cd8-678f-481b-9060-29989ba741a7" />


```
loss_function = nn.MSELoss()  # Mean Squared Error (MSE) loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent

```



```
epochs = 50  # Number of training iterations
losses = []  # List to store loss values

for epoch in range(1, epochs + 1):  # Start from 1 to 50
    optimizer.zero_grad()  # Clear previous gradients
    y_pred = model(X)  # Forward pass
    loss = loss_function(y_pred, y)  # Compute loss
    losses.append(loss.item())  # Store loss value
    
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Print loss, weight, and bias for EVERY epoch (1 to 50)
    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')

```
#### output 4
<img width="586" alt="Screenshot 2025-03-06 at 2 48 56 PM" src="https://github.com/user-attachments/assets/f4e3a372-a523-4933-b0e3-958b6f47cfc1" />


```
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');
plt.show()
```
#### output 5
<img width="219" alt="Screenshot 2025-02-27 at 2 16 29 PM" src="https://github.com/user-attachments/assets/160d6456-c3a0-4ae1-911a-f5192f1a0187" />




```
# Automatically determine x-range
x1 = torch.tensor([X.min().item(), X.max().item()])

# Extract model parameters
w1, b1 = model.linear.weight.item(), model.linear.bias.item()

# Compute y1 (predicted values)
y1 = x1 * w1 + b1

```


```
# Print weight, bias, and x/y values
print(f'Final Weight: {w1:.8f}, Final Bias: {b1:.8f}')
print(f'X range: {x1.numpy()}')
print(f'Predicted Y values: {y1.numpy()}')
```
#### output 7

<img width="446" alt="Screenshot 2025-03-07 at 8 36 08 AM" src="https://github.com/user-attachments/assets/5776f767-ae1f-4e46-9d22-894c5612b563" />


```
# Plot original data and best-fit line
plt.scatter(X.numpy(), y.numpy(), label="Original Data")
plt.plot(x1.numpy(), y1.numpy(), 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
```
#### output 8
<img width="570" alt="Screenshot 2025-03-07 at 8 36 41 AM" src="https://github.com/user-attachments/assets/b7647a4e-7519-48d7-b87a-34d3af01d076" />





### RESULT 
#### The trained model successfully approximated the linear relationship with Final Weight ≈ 2.0 and Final Bias ≈ 1.0. The loss reduced over epochs, and the best-fit line closely followed the original data distribution, confirming successful training.

