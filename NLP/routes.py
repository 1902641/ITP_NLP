from flask import render_template, url_for, flash, redirect, request, abort, send_from_directory, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
from NLP import app, pdfManagement, csvReader
import flask_excel as excel
import urllib.request, json
import os

ALLOWED_EXTENSIONS = set(['pdf'])





def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

labelsList = csvReader.readCSV()

@app.route("/")
def home():
	return render_template("home.html")


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


@app.route('/index_get_data')
def stuff():
  # Assume data comes from somewhere else
  data = {
    "data": [
      {
        "PDF_Name": "Door",
		"Confidence_Level": "50%"
      },
	  {
		"PDF_Name": "Fire Resistance Door",
		"Confidence_Level": "30%"
      },
	  {
		"PDF_Name": "Fire Resistance Bed",
		"Confidence_Level": "29%"
      },
	  {
		"PDF_Name": "Wooden Bedframe",
		"Confidence_Level": "8%"
      },
	  {
		"PDF_Name": "Bed",
		"Confidence_Level": "4%"
      },
	  {
		"PDF_Name": "Water Resistance Bed",
		"Confidence_Level": "2%"
      }]
  }
  return jsonify(data)


@app.route('/upload_get_data')
def uploadData():
  # Assume data comes from somewhere else
  data = {
    "data": [
      {
		"PDF_Name" : "Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "80%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Search.pdf",
		"Label_Attached": "Search",
		"Confidence_Level": "67%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      },
	  {
		"PDF_Name" : "Fire Resistance Door.pdf",
		"Label_Attached": "Door",
		"Confidence_Level": "88%",
		"DateOfUpload": "6/6/2021",
		"ManualCheck": "True"
      }]
  }
  return jsonify(data)


@app.route("/view")
def view():
	return render_template("view.html", title="View Documents")