import io
import torch
import torch.nn as nn

import torchvision.transforms as transforms

from PIL import Image
import base64
import numpy as np

import flask
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r'/*' : {'origins': '*'}})


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

device = torch.device('cuda:0')
CNN_MODEL = CNN().to(device)
CNN_MODEL.load_state_dict(torch.load('./cnn_model.pth', map_location=device))
CNN_MODEL.eval()



@app.route('/predict', methods=['POST'])

def predict():
    
    if request.method == "POST":

        image = request.json['image']
        image_string = base64.b64decode(image)
        
        image = Image.open(io.BytesIO(image_string))
        image = image.convert('1') #convert to BW
        image = image.resize((28,28)) #resize the image

        image_torch = torch.tensor(np.float32(np.array(image))).to(device)

        output = CNN_MODEL(image_torch.view(-1, 1, 28,28))
	
        softmax_out = nn.Softmax(dim=1)(output)

        top_pred = torch.topk(softmax_out[0], 2)[1]

        response = flask.jsonify({"predictions":top_pred.tolist()})
        

        return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
