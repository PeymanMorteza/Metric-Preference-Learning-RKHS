import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cpu")
#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Example data and labels
#X_np = np.array([[1, 0, -1, 0]], dtype=np.float32)  # Shape (n, 2d)
#Y_np = np.array([-1], dtype=np.float32)  # Labels corresponding to each data point
# X_np=X_tr
# Y_np=Y_tr
# X = torch.tensor(X_np, requires_grad=False)
# Y = torch.tensor(Y_np, requires_grad=False)

# n, d2 = X.shape
# d = d2 // 2

# # Initialize lower triangular matrix B
# B = torch.randn(d, d, dtype=torch.float64, requires_grad=True)
# u = torch.randn(d, dtype=torch.float64, requires_grad=True)

# Define the PSD matrix A as B^T B
def compute_psd_matrix(B):
    return B @ B.T

# Ensure A is symmetric by forcing it to be symmetric in each optimization step
def make_symmetric(mat):
    return (mat + mat.T) / 2

# Define the hinge loss function
def hinge_loss(prediction, y):
    # Convert y to the range of {-1, 1} if needed
    #y = 2 * y - 1  # Converts labels {0, 1} to {-1, 1}
    return torch.max(torch.tensor(0.0), 1 - y * prediction)
# Define the hinge loss function
def zero_one_loss(prediction, y):

    return int(int(torch.sign(prediction))!=int(y.item()))

# Define the loss function
def loss_function(B, u, x, y,lambda_reg=0,eval_func=hinge_loss):
    d,_=B.shape
    # Compute PSD matrix A from B
    A = compute_psd_matrix(B)
    
    # Split x into x1 and x2
    x1 = u - x[:d]
    x2 = u - x[d:]
    
    # Compute the quadratic forms
    A = make_symmetric(A)  # Ensure A is symmetric
    term1 = x1 @ (A @ x1)
    term2 = x2 @ (A @ x2)
    
    # Compute the prediction
    prediction = term1 - term2
    
    # Compute hinge loss+ reg term required by our repr thm
    #print("prediction",type(prediction))
    #print("y",type(y))
    #print("p",prediction)
    #print("sign",int(torch.sign(prediction)))
    #print("yy",int(y.item()))
    #print("res",y.item(),torch.sign(prediction),int(torch.sign(prediction))==int(y.item()))




    return eval_func(prediction, y)+lambda_reg*(u @ (A @ u))

def train(X_np, Y_np, num_epochs=1000, loss_fn=loss_function,reg_lam=0.0,batch_size=32):
    """
    Train a model using gradient descent to optimize the matrix B and vector u.

    Args:
    - X_np (numpy.ndarray): Input data of shape (n, d2), where n is the number of data points and d2 is the number of features (twice the dimension of the original space).
    - Y_np (numpy.ndarray): Labels for the data of shape (n,), where each element is the label for the corresponding data point in X_np.
    - num_epochs (int, optional): Number of epochs (iterations) for training. Default is 1000.
    - loss_fn (function, optional): Loss function to compute the loss between the model's predictions and the true labels. It should take as inputs the matrix B, vector u, a data point x_i, and a label y_i. Default is `loss_function`.

    Returns:
    - A_opt (numpy.ndarray): The optimal positive semi-definite matrix A of shape (d, d), computed from the trained matrix B.
    - u_opt (numpy.ndarray): The optimal vector u of shape (d,), obtained from the training process.
    """

    # Convert input data and labels from numpy arrays to PyTorch tensors
    #X = torch.tensor(X_np, dtype=torch.float32, requires_grad=False)
    #Y = torch.tensor(Y_np, dtype=torch.float32, requires_grad=False)

    X = torch.tensor(X_np, dtype=torch.float32, requires_grad=False).to(device)
    Y = torch.tensor(Y_np, dtype=torch.float32, requires_grad=False).to(device)
    dataset = TensorDataset(X, Y)
    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Determine the number of data points (n) and the feature dimension (d2)
    n, d2 = X.shape
    #print("n",n)
    #print("d2",d2)
    # Compute the original dimension (d) from d2
    d = d2 // 2

    # Initialize matrix B as a random lower triangular matrix of shape (d, d)
    #B = torch.randn(d, d, dtype=torch.float32, requires_grad=True)
    # Initialize vector u as a random vector of shape (d,)
    #u = torch.randn(d, dtype=torch.float32, requires_grad=True)

    B = torch.randn(d, d, dtype=torch.float32, requires_grad=True, device=device)
    u = torch.randn(d, dtype=torch.float32, requires_grad=True, device=device)
    
    # Define the optimizer to update B and u
    optimizer = torch.optim.Adam([B, u], lr=0.01)
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            x_batch, y_batch = batch
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Compute loss for the batch
            batch_loss = 0
            for i in range(x_batch.shape[0]):
                x_i = x_batch[i]
                y_i = y_batch[i]
                loss = loss_fn(B, u, x_i, y_i, lambda_reg=reg_lam)
                batch_loss += loss
            
            # Average the batch loss
            batch_loss /= x_batch.shape[0]
            total_loss += batch_loss.item()
            
            # Backpropagation and optimization
            batch_loss.backward()
            optimizer.step()
        
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            #print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')
            logging.info(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')
    # # Optimization loop
    # for epoch in range(num_epochs):
    #     # Zero the gradients for this iteration
    #     optimizer.zero_grad()
        
    #     total_loss = 0
    #     # Compute the total loss over all data points
    #     for i in range(n):
    #         # Extract the i-th data point and its label
    #         x_i = X[i, :]
    #         y_i = Y[i]
    #         # Compute the loss for the current data point
    #         loss = loss_fn(B, u, x_i, y_i,lambda_reg=reg_lam)
    #         # Accumulate the total loss
    #         total_loss += loss
        
    #     # Average the total loss over all data points
    #     total_loss /= n
        
    #     # Compute gradients for the parameters
    #     total_loss.backward()
    #     # Update parameters B and u
    #     optimizer.step()
        
    #     # Print the loss at every 100 epochs
    #     if epoch % 100 == 0:
    #         print(f'Epoch {epoch}, Loss: {total_loss.item()}')

    # Compute the optimal matrix A from the final B
    #A_opt = compute_psd_matrix(B).detach().numpy()
    # Convert the final vector u to a NumPy array
    #u_opt = u.detach().numpy()

    A_opt = compute_psd_matrix(B).detach().cpu().numpy()  # Move back to CPU for NumPy compatibility
    u_opt = u.detach().cpu().numpy()
    
    return A_opt, u_opt
def evaluate_on_test_data(B, u, x_test, y_test):
    """
    Evaluate the loss on test data.
    
    Parameters:
        B (torch.Tensor): Matrix B used to compute the PSD matrix A.
        u (torch.Tensor): Vector u used in the loss function.
        x_test (torch.Tensor): Test data features. Should be of shape (n_samples, 2*d).
        y_test (torch.Tensor): Test data labels. Should be of shape (n_samples,).
        d (int): Dimensionality of the split feature vectors x1 and x2.
        
    Returns:
        total_loss (torch.Tensor): Computed loss on the test data.
    """
    # x_test=torch.tensor(x_test,dtype=torch.float32)
    # y_test=torch.tensor(y_test,dtype=torch.float32)
    # B=torch.tensor(B,dtype=torch.float32)
    # u=torch.tensor(u,dtype=torch.float32)

    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    B = torch.tensor(B, dtype=torch.float32).to(device)
    u = torch.tensor(u, dtype=torch.float32).to(device)
    
    total_loss = 0.0
    for i in range(x_test.shape[0]):
        x = x_test[i]
        y = y_test[i]
        #print(x)
        #print(y)
        loss = loss_function(B, u, x, y,eval_func=zero_one_loss,lambda_reg=0.0)
        #print("x",x)
        #print("y",y)
        #print("B",B)
        #print("u",u)
        #print("loss",loss)
        total_loss += loss
    
    # Average loss over all test samples
    average_loss = total_loss / x_test.shape[0]
    
    return average_loss


