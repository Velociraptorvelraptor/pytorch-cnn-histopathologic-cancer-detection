import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify

from model import CNNImageClassifier

CHECKPOINT_PATH = 'checkpoints/cancer-epoch=67-train_loss=0.26.ckpt'

model = CNNImageClassifier.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

IMAGE_SIZE = 32


def transform_image(img):
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)


def get_prediction(img):
    result = model(img)
    return F.softmax(result, dim=1)[:, 1].tolist()[0]


app = Flask(__name__)

@app.route('/')
def hello():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST'])
def predict():
    img_file = request.files['image']
    img = Image.open(img_file.stream)
    img = transform_image(img)
    prediction = get_prediction(img)
    if prediction > 0.5:
        output = "true"
    else:
        output = "false"
    return jsonify({'cancer_or_not': output})



if __name__ == "__main__":
    app.run(debug=True)