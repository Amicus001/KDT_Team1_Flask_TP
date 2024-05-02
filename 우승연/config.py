# 모듈 로딩
import os

# 다양한 DBMS URI - SQLITE
BASE_DIR = os.path.dirname(__file__)

# 
DB_USER_ID = 'root'
DB_USER_PW = 'kdt5'
DB_HOST_IP = '1.251.203.204'
DB_NAME = 'Team1'
DB_CHARSET = 'utf-8'

## 다양한 DBMS URI
# DB_SQLITE_URI = f'sqlite:///{os.path.join(BASE_DIR, DB_NAME_SQLITE)}'
DB_MYSQL_URI = 'mysql+pymysql://root:kdt5@1.251.203.204:33065/Team1'


## 사용할 DBMS 설정
SQLALCHEMY_DATABASE_URI = DB_MYSQL_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False