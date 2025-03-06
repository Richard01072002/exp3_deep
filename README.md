# EXP-3
### Name- RICHARDSON A
### reg no. - 212222233005
### dep - AI & DS 
## AIM
#### To understand and demonstrate the conversion of NumPy arrays into PyTorch tensors and explore various ways to create tensors in PyTorch.
## procedure 
#### Step 1: Install OpenCV
#### Step 2: Check PyTorch Version
#### Step 3: Create a NumPy array and print its type and data type.
#### Step 4: Create a 2D NumPy array and reshape it and Convert it to a PyTorch tensor and print the results.
#### Step 5: Modify the original NumPy array and observe changes in the tensor when using torch.from_numpy().
#### Step 6: Create PyTorch Tensors Using Different Methods:
#### Step 7: Use torch.Tensor(data) to create a tensor. Use torch.tensor(data) to create a tensor with the default type.
#### Step 8: Create an uninitialized tensor with torch.empty().Create a zero tensor using torch.zeros().
#### Step 9: Generate a tensor with evenly spaced values using torch.arange().Create a tensor using torch.linspace() for equally spaced values.

## program
```
iimport torch
import torch.nn as nn  # Neural network module

import numpy as np
import matplotlib.pyplot as plt  # For plotting
%matplotlib inline  
```
#### output 1
<img width="166" alt="Screenshot 2025-02-27 at 2 11 30 PM" src="https://github.com/user-attachments/assets/261eb803-aed0-4f74-ae1d-58876bdf8b0b" />



```
X = torch.linspace(1,50,50).reshape(-1,1)
```
#### output 2
<img width="212" alt="Screenshot 2025-02-27 at 2 12 21 PM" src="https://github.com/user-attachments/assets/08373bfa-0076-459e-aab3-300593d333b3" />




```
torch.manual_seed(71) # to obtain reproducible results
e = torch.randint(-8,9,(50,1),dtype=torch.float)
```
#### output 3
<img width="162" alt="Screenshot 2025-02-27 at 2 12 42 PM" src="https://github.com/user-attachments/assets/ca65f180-7bf3-4074-9aca-4c4b774096de" />




```
y=2*X+1+e
print(y.shape)
```
#### output 4
<img width="146" alt="Screenshot 2025-02-27 at 2 13 03 PM" src="https://github.com/user-attachments/assets/9182e13a-0633-42bc-a0aa-f27f39befca9" />


```
plt.scatter(X.numpy(), y.numpy(),color='red')  # Scatter plot of data points
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
```
#### output 5
<img width="218" alt="Screenshot 2025-02-27 at 2 13 34 PM" src="https://github.com/user-attachments/assets/844f8e43-b111-407b-be3f-a0f7ad634a73" />


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
#### output 6
<img width="172" alt="Screenshot 2025-02-27 at 2 13 57 PM" src="https://github.com/user-attachments/assets/385dc7aa-58fe-4580-b3e4-3997b695ee84" />


```
# Creating an instance of the model
torch.manual_seed(59)
model = Model(1, 1)
print('Weight:', model.linear.weight.item())
print('Bias:  ', model.linear.bias.item())
```
#### output 7
<img width="421" alt="Screenshot 2025-02-27 at 2 14 35 PM" src="https://github.com/user-attachments/assets/bd88cfdc-8472-46cc-b664-9174468ab333" />


```
loss_function = nn.MSELoss()  # Mean Squared Error (MSE) loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent

```
#### output 8
<img width="290" alt="Screenshot 2025-02-27 at 2 15 23 PM" src="https://github.com/user-attachments/assets/b5ca9f3c-6fd7-4dba-b284-b37c46000aa9" />



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
#### output 9
<img width="288" alt="Screenshot 2025-02-27 at 2 15 49 PM" src="https://github.com/user-attachments/assets/fb05a1a2-75ac-44b5-8009-85eab3a1b69d" />



```
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');
plt.show()
```
#### output 10
<img width="219" alt="Screenshot 2025-02-27 at 2 16 29 PM" src="https://github.com/user-attachments/assets/160d6456-c3a0-4ae1-911a-f5192f1a0187" />




```
# Automatically determine x-range
x1 = torch.tensor([X.min().item(), X.max().item()])

# Extract model parameters
w1, b1 = model.linear.weight.item(), model.linear.bias.item()

# Compute y1 (predicted values)
y1 = x1 * w1 + b1

```
#### output 11
<img width="234" alt="Screenshot 2025-02-27 at 2 16 44 PM" src="https://github.com/user-attachments/assets/e07b884c-910e-4437-920c-78b939ee4723" />




```
# Print weight, bias, and x/y values
print(f'Final Weight: {w1:.8f}, Final Bias: {b1:.8f}')
print(f'X range: {x1.numpy()}')
print(f'Predicted Y values: {y1.numpy()}')
```
#### output 13
<img width="369" alt="Screenshot 2025-02-27 at 2 17 32 PM" src="https://github.com/user-attachments/assets/43ba2ddc-935c-4dba-8b2a-58c9714c4799" />


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
#### output 14
<img width="338" alt="Screenshot 2025-02-27 at 2 18 07 PM" src="https://github.com/user-attachments/assets/eac40b69-b11b-4513-92d1-fd64503abe05" />





### Result 
#### Successfully created and manipulated PyTorch tensors from NumPy arrays. Understood the difference between torch.tensor() and torch.from_numpy(). Learned how PyTorch tensors share memory with NumPy arrays when using from_numpy(). Demonstrated various ways to create tensors using PyTorch functions like empty(), zeros(), arange(), and linspace().


