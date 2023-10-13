import pandas as pd
import numpy as np

class SVM:
    def __init__(self, train_dataset, test_dataset) -> None:
        """Construtor do modelo SVM (Support Vector Machine)

        Args:
            train_dataset (_type_): _description_
            test_dataset (_type_): _description_
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dim_feature_space = train_dataset.shape[1] - 1
        self.weights = np.zeros(self.dim_feature_space)
        self.b_coef = 0
    
    
    def predict(self, input: np.array) -> bool:
        """Prediz a classe correspondente a uma certa instância x

        Args:
            input (np.array): o vetor de entrada x = (x_1, x_2, x_3, ..., x_n)

        Returns:
            bool: True se x é da classe positiva e False caso contrário
        """
        # f(x) = w^Tx + b
        discriminant = np.dot(input, self.weights) + self.b_coef
        
        if discriminant >= 0:
            # f(x) >= 0 (classe positiva)
            return True
        else:
            # f(x) < 0 (classe negativa)
            return False