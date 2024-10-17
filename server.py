from flask import Flask, render_template, request
import cv2
import numpy as np
import json
import os
from werkzeug.utils import secure_filename
from plyfile import PlyData, PlyElement
from infer import yolo_model

from datetime import datetime


model = yolo_model()
app = Flask(__name__, template_folder='template')
img = os.path.join('static', 'image')


@app.route('/', methods=['GET', 'POST'])
def index():
    no_image_display = True
    if request.method == 'POST':

        if len(request.files.getlist("file")) > 0:
            # file = request.files['file']
            uploaded_files = request.files.getlist("file")
            now = datetime.now()  # current date and time
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            # date_time = r'{}'.format(date_time_g)
            os.mkdir(date_time)
            number_of_file = len(uploaded_files)
            print('there are:', number_of_file)
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                file.save(filename)
                model.run_model(number_of_file == 1, filename,
                                filename[:-4], date_time)
            if (number_of_file == 1):
                no_image_display = False

    if (no_image_display):
        return render_template('index.html')
    else:
        file = os.path.join(img, 'detected_frame.png')
        return render_template('index.html', image=file)


if (__name__ == '__main__'):
    app.run()
