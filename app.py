import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load your model
model = LogisticRegression(input_dim=28*28, output_dim=10)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

app = FastAPI()

# Define the transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Ensure the image is resized to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create a simple HTML form for file upload
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Upload</title>
</head>
<body>
    <h1>Upload an Image for Prediction</h1>
    <form action="/predict/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')
    image = transform(image).unsqueeze(0)  # No need to flatten here
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return {"prediction": predicted.item()}

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        import asyncio
        asyncio.run(uvicorn.run(app, host="127.0.0.1", port=8000))
