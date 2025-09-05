# Champlain Hackathon 2025 - Trash Classifier 🗑️🤖

Welcome to our Champlain Hackathon 2025 project! This app uses **deep learning** to classify images of waste into **organic** or **recycling** categories. Perfect for building smarter recycling solutions! 🌱♻️

---

## Features ✨

- **Image Classification:** Upload or provide a URL to an image of trash.
- **Real-time Predictions:** Get results instantly via a Flask API.
- **Confidence Scores:** See how confident the model is about its prediction.
- **Easy Integration:** Use the `classify_image()` function directly in Python.

---

## How It Works 🛠️

1. The model is a custom **LeNet CNN** built with PyTorch.
2. Images are resized to 128x128, normalized, and passed through the model.
3. The model outputs probabilities for each class: `organic` or `recycling`.
4. Results include **predicted class** and **confidence**.

---

## Getting Started 🚀

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- Flask
- Pillow
- requests

### Install Dependencies

```bash
pip install torch torchvision flask pillow requests flask-cors
```

### Running the App

```bash
python backend.py
```

Open your browser at `http://localhost:65535` to access the web interface.

### API Usage

Send a POST request to `/predict` with:

- **image** (file upload)
- **url** (image URL)
- **image\_data** (base64 string)

Example using Python requests:

```python
import requests

url = 'http://localhost:65535/predict'
files = {'image': open('trash.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

---

## Direct Python Usage 🐍

```python
from backend import classify_image

result = classify_image('trash.jpg')
print(result)
```

---

## Classes

- `organic` 🌿
- `recycling` ♻️

---

**Have fun sorting trash with AI!** 😎

