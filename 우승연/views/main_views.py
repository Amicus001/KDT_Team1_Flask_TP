from flask import Blueprint, render_template
from flask import Blueprint, render_template, request
from ..models import db
from datetime import datetime
import torch
# from CustomClass import TrashClassifier

from PIL import Image, ImageFile
from ..models import Trashlist  # dbmodel.py에서 테이블 객체 임포트


bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def general():
    return render_template('index.html')

@bp.route('/trash_image_submitted')
def submitted():
    # 현재 시간과 쓰레기 타입을 지정하여 Trashlist 테이블에 데이터 추가
    col = Trashlist(type="some_type", create_date=datetime.now())  # type은 임시로 "some_type"으로 설정
    db.session.add(col)  # 행 추가
    db.session.commit()  # 변경사항 저장

    return render_template('index.html')

