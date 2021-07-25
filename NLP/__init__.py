from flask import Flask
import os
from flask_dropzone import Dropzone

app = Flask(__name__)
dropzone = Dropzone(app)
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = '.pdf'
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.secret_key = "secret key"
filename = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'static' ,'uploads'))
UPLOAD_FOLDER = filename
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 2048 * 2048
app.config['TRAIN_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'static' ,'trains'))

from NLP import routes