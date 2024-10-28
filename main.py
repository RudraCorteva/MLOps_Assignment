import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import sys
import uvicorn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = LogisticRegression(input_dim=28*28, output_dim=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

app = FastAPI()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Ensure the image is resized to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')
    image = transform(image).unsqueeze(0).view(-1, 28*28)  # Flatten the image
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return {"prediction": predicted.item()}

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        import asyncio
        asyncio.run(uvicorn.run(app, host="0.0.0.0", port=8000))
