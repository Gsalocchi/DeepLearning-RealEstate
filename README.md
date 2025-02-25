# DeepLearning-RealEstate
# RealEstateNN

RealEstateNN is a PyTorch-based neural network designed for predicting real estate values. The model leverages a series of fully connected layers with batch normalization, the SiLU (Swish) activation function, and dropout regularization to effectively learn non-linear relationships from input features.

## Model Architecture

The network is built using a sequential model with the following structure:

1. **Input Layer:**  
   The network accepts an input vector of size `input_size` (which represents the number of features in your dataset).

2. **Hidden Layers:**  
   - **First Hidden Layer:**  
     - Linear transformation from `input_size` to 54 units.
     - Batch Normalization on the 54 units.
     - SiLU activation (also known as Swish).
     - Dropout with a probability of 0.3 to prevent overfitting.
     
   - **Second Hidden Layer:**  
     - Linear transformation from 54 to 48 units.
     - Batch Normalization on the 48 units.
     - SiLU activation.
     - Dropout with a probability of 0.3.
     
   - **Third Hidden Layer:**  
     - Linear transformation from 48 to 24 units.
     - Batch Normalization on the 24 units.
     - SiLU activation.
     
   - **Fourth Hidden Layer:**  
     - Linear transformation from 24 to 12 units.
     - Batch Normalization on the 12 units.
     - SiLU activation.

3. **Output Layer:**  
   - A linear transformation from 12 units to a single output, representing the predicted real estate value.

## Code Example

Below is the complete code for the `RealEstateNN` model:

```python
import torch
import torch.nn as nn

class RealEstateNN(nn.Module):
    def __init__(self, input_size):
        super(RealEstateNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 54),
            nn.BatchNorm1d(54),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Linear(54, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.SiLU(),

            nn.Linear(24, 12),
            nn.BatchNorm1d(12),
            nn.SiLU(),

            nn.Linear(12, 1)
        )

    def forward(self, x):
        return self.model(x)
```

## How It Works

- **Linear Layers:**  
  Each linear layer transforms the input into a higher-level representation. The gradual reduction in the number of units helps the network learn a compressed representation of the input features.

- **Batch Normalization:**  
  Batch normalization is applied after each linear layer (except the output) to stabilize and speed up training by normalizing the activations.

- **SiLU Activation:**  
  The SiLU (Sigmoid Linear Unit) activation function, equivalent to the swish function, introduces non-linearity and helps the model learn complex patterns.

- **Dropout:**  
  Dropout layers with a probability of 0.3 are used after the first two hidden layers to reduce overfitting by randomly dropping a fraction of the units during training.

## Usage

To use the `RealEstateNN` model in your project:

1. **Instantiate the Model:**

   ```python
   input_size = 10  # Set this to the number of features in your dataset
   model = RealEstateNN(input_size)
   ```

2. **Forward Pass:**

   ```python
   sample_input = torch.randn(1, input_size)  # Example input tensor
   output = model(sample_input)
   print(output)
   ```

3. **Training:**

   Use standard PyTorch training loops with an optimizer and loss function of your choice to train the model on your real estate dataset.

## Conclusion

The RealEstateNN model is a robust and flexible architecture for real estate value prediction. By combining linear transformations, batch normalization, the SiLU activation, and dropout regularization, it is well-suited to capture complex relationships in the data and generalize well to unseen examples.

Feel free to contribute, open issues, or improve the model further. Happy coding!

--- 

*This project is intended for educational and research purposes.*
