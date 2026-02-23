# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
1.Load and Preprocess Data
2.Get the shape of the first image in the training dataset
3.Get the shape of the first image in the test dataset
4.Train the Model
5.Test the Model
6.Predict on a Single Image
7.Display the image 





## PROGRAM

### Name:J.JANANI

### Register Number: 212223230085

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the Model
# Step 3: Train the Model

def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Print once per epoch (correct indentation)
        print("Name: J.JANANI")
        print("Register Number: 212223230085")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

```

### OUTPUT

## Training Loss per Epoch

<img width="332" height="213" alt="image" src="https://github.com/user-attachments/assets/a7efd69e-71a5-4cf5-8fbe-3e5253dba2e2" />


## Confusion Matrix

<img width="325" height="132" alt="image" src="https://github.com/user-attachments/assets/25d79857-45cc-44ee-90db-d0515169a426" />
<img width="962" height="722" alt="image" src="https://github.com/user-attachments/assets/74414e3e-1b92-4a22-9297-4ba377c7a77f" />



## Classification Report
<img width="590" height="440" alt="image" src="https://github.com/user-attachments/assets/028e4b93-08fe-4789-867b-f1052bf70674" />

### New Sample Data Prediction
<img width="623" height="622" alt="image" src="https://github.com/user-attachments/assets/e3d50fba-833f-4a29-abc7-1ed5460b39a5" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
