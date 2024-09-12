import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
# import onnxsim
# import onnx

from models.model import DeepLab
from utils.initialization import cvtColor, preprocess_input, resize_image, show_config

class DeeplabV3:
    _defaults = {
        "model_path": 'logs/best_epoch_weights.pth',
        "num_classes": 19,
        "backbone": "mobilenet",
        "input_shape": [512, 512],
        "downsample_factor": 16,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        # Update default settings with any user-defined settings
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # Define colors for different classes
        self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
                       (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), 
                       (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), 
                       (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0)]
        
        # Load the model
        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        # Load model and weights
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone, downsample_factor=self.downsample_factor, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.eval()
        print(f'{self.model_path} model, and classes loaded.')
        
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    def detect_image(self, image):
        # Preprocess image
        image = cvtColor(image)
        orininal_h, orininal_w = image.size
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).cuda() if self.cuda else torch.from_numpy(image_data)
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[(self.input_shape[0] - nh) // 2:(self.input_shape[0] - nh) // 2 + nh, 
                    (self.input_shape[1] - nw) // 2:(self.input_shape[1] - nw) // 2 + nw]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR).argmax(axis=-1)
        return pr

    def get_FPS(self, image, test_interval):
        # Preprocess image
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).cuda() if self.cuda else torch.from_numpy(image_data)

        # Measure FPS
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[(self.input_shape[0] - nh) // 2:(self.input_shape[0] - nh) // 2 + nh, 
                        (self.input_shape[1] - nw) // 2:(self.input_shape[1] - nw) // 2 + nw]
        t2 = time.time()
        return (t2 - t1) / test_interval

    def convert_to_onnx(self, simplify, model_path):
        self.generate(onnx=True)
        dummy_input = torch.zeros(1, 3, *self.input_shape).to('cpu')
        
        # Export the model to ONNX format
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net, dummy_input, model_path, opset_version=12, do_constant_folding=True, 
                          input_names=["images"], output_names=["output"])

        # Verify and simplify the ONNX model
        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'Simplification check failed'
            onnx.save(model_onnx, model_path)

        print(f'ONNX model saved as {model_path}')
    
    def get_miou_png(self, image):
        # Preprocess image
        image = cvtColor(image)
        orininal_h, orininal_w = image.size
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).cuda() if self.cuda else torch.from_numpy(image_data)
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[(self.input_shape[0] - nh) // 2:(self.input_shape[0] - nh) // 2 + nh, 
                    (self.input_shape[1] - nw) // 2:(self.input_shape[1] - nw) // 2 + nw]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR).argmax(axis=-1)
    
        return Image.fromarray(np.uint8(pr))
