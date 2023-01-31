import numpy as np
import torch

class LinearRegression:
    """ Linear Regression model from scratch
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Linear Regresssion model fitting

        Args:
            X (np.ndarray): Input training data 
            y (np.ndarray): Input training output.
        """
        # Check if array has a determinant
        if np.linalg.det(X.T @ X)!=0:
            X_train_append= np.hstack((X, np.ones((X.shape[0],1))))
            wb = np.linalg.inv(X_train_append.T@X_train_append)@X_train_append.T@y
            
            self.w = wb[:-1]
            self.b = wb[-1]

        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predicts out out put using previously calculated weights using linear regression.

        Args:
            X (np.ndarray): Input data to predict.
        Returns:
            prediction (np.ndarray): Predicted values.
        """        
        prediction = X @ self.w + self.b

        return(prediction)

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """


    def _squared_error(self, yhat, y):
        '''
        Squared error loss function.
        
        Args:
            y_hat (torch.tensor): Predicted values.
            y (torch.tensor): True values.
        
        Returns:
            err (FLOAT): The squared error (loss).
        
        '''
        err = (yhat - y.reshape(yhat.shape)) ** 2
        return err
    
    def _gradient_descent(self, w, b, lr):
        '''
        Gradient_descent algorithm.
        
        Args:
            w (torch.tensor): Model weights.
            b : torch.tensor: Bias.
            lr (FLOAT): learning rate.
        
        Returns:
        w, b.
        
        '''
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            # Set gradient to zero to flush the cache
            w.grad.zero_()
            b.grad.zero_()

        return (w, b)


    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Gradient descent fitting of weights.

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
            lr (float, optional): _description_. Defaults to 0.01.
            epochs (int, optional): _description_. Defaults to 1000.
        """

        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y.squeeze(), dtype=torch.float32)

        w = torch.zeros(X.shape[1], 1 , dtype=torch.float32, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

        for e in range(epochs):
            # Fit the model to the training data:
            yhat = X_train @ w + b
            l = self._squared_error(yhat, y_train).mean()
            l.backward()

            w, b = self._gradient_descent(w, b, lr)
        self.w = w
        self.b = b

        # raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            pred_np (np.ndarray): The predicted output.

        """

        prediction = torch.tensor(X, dtype=torch.float32) @ self.w + self.b
        pred_np = prediction.detach().numpy()
        
        return(pred_np)
