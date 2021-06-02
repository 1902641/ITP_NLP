from flask import render_template, url_for, flash, redirect, request, abort, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from NLP import app, pdfManagement
import urllib.request
import os

ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template(
        "home.html",
    )

@app.route('/upload')
def upload_form():
	pdfManagement.retrieveListOfPDF()
	return render_template('upload.html')

@app.route('/uploading', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the files part
		if 'files[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = request.files.getlist('files[]')
		for file in files:
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('File(s) successfully uploaded')
		return redirect('/upload')

@app.route("/label")
def label():
    return render_template(
        "label.html",
        title="Label Documents",
    )