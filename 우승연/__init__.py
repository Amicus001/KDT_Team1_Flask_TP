from flask import Flask, request, redirect, render_template, url_for
from Trash.views import main_views
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import io
import os
from torchvision.models import mobilenet_v2
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

#DB instance------------------------------------------------------------
db = SQLAlchemy()
migrate = Migrate()
################################################################

def load_model(model_path):
    model = mobilenet_v2(pretrained=False)
    
    state_dict = torch.load(model_path)
    in_features = model.classifier[1].in_features
    out_features = state_dict['classifier.1.weight'].shape[0]
    
    model.classifier[1] = torch.nn.Linear(in_features, out_features)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def preprocess_image(image):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict_image(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


def create_app():
    app = Flask(__name__)

    MODEL_PATH = 'Model2.pth'
    model = load_model(MODEL_PATH)

    #설정 내용 로딩
    app.config.from_pyfile('config.py')

    #ORM (DB 초기화)
    db.init_app(app)
    migrate.init_app(app, db)


    @app.route('/predict', methods=['POST', 'GET'])
    def predict():
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('general.html', result=None, image_url=None)

            file = request.files['file']
            if file.filename == '':
                return render_template('general.html', result=None, image_url=None)

            if file:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_tensor = preprocess_image(image)

                prediction = predict_image(model, image_tensor)
                result = 0 if prediction == 0 else 1

                # 제출된 이미지 서버에 저장
                image_path = os.path.join(app.root_path, 'static', "submitted_images.jpg")
                image.save(image_path)

                image_url = url_for('static', filename="submitted_images.jpg")

                return render_template('index.html', result=result, image_url=image_url)

        return render_template('index.html')

    # Blueprint 등록
    app.register_blueprint(main_views.bp)

    return app

