# Stock-Price-Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.

<img width="1092" height="380" alt="image" src="https://github.com/user-attachments/assets/00055ce3-3c0e-46f3-8475-ece6f0336013" />


## Design Steps

### Step 1:
Import necessary libraries.
### Step 2:
Load and preprocess the data.
### Step 3:
Create input-output sequences.
### Step 4:
Convert data to PyTorch tensors.
### Step 5:
Define the RNN model.
### Step 6:
Train the model using the training data.
### Step 7:
Evaluate the model and plot predictions.

## Program
#### Name: JISHA BOSSNE SJ
#### Register Number:212224230106

```
# Define RNN Model

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
```
# Train the Model

num_epochs = 20
train_losses = []

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
       
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="403" height="365" alt="image" src="https://github.com/user-attachments/assets/e1750a54-3440-4659-8843-d5597cc368cf" />

<img width="753" height="507" alt="image" src="https://github.com/user-attachments/assets/5bbf129d-e9c8-4cdc-bab2-ecb0e0e4bf19" />

### Predictions 

<img width="1070" height="641" alt="image" src="https://github.com/user-attachments/assets/3ca85365-e3f5-467d-a780-d8b6984acf7a" />

## Result

The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.


