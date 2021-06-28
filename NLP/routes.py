from flask import render_template, url_for, flash, redirect, request, abort, send_from_directory, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
from NLP import app, pdfManagement, csvReader
import flask_excel as excel
import urllib.request, json
import os
from NLP.nlp_model.BERTModel import BERTModel
import pandas as pd

ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


labelsList = csvReader.readCSV()

@app.route("/")
def home():
	return render_template("home.html", labelsList=labelsList)


@app.route('/upload')
def upload_form():
	pdfManagement.retrieveListOfPDF()
	return render_template('upload.html', labelsList=labelsList)


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

		# -------------------------------------------------------------------
		# # Assuming this is where text extraction and prediction will be done
		# # Makeshift label list loader
		# label_file = open("label.txt", "r")
		# label_list = label_file.read().splitlines()

		# # Load the labels first before loading model
		# bert_model = BERTModel()
		# bert_model.load_label_list(label_list)

		# # Optional - load or save to different directory
		# # If loading different models, call this before load_model()
		# new_directory = "./bert_model_v2"
		# bert_model.set_output_model_directory(new_directory)
		# bert_model.output_model_directory(new_directory)

		# Call load model if configuration are done
		# bert_model.load_model()

		# # Single Prediction
		# single_string = "Single string to test"
		# # Clean text for model to predict
		# single_string = bert_model.clean_text(single_string)
		# bert_model.predict(single_string, single_prediction=True)

		# Batch Prediction - for a list of predictions to perform
		# batch_string = [single_string, single_string, single_string]
		# # As always, clean text for model to predict
		# batch_string = list(map(bert_model.clean_text, batch_string))
		# # By default, single prediction is set to false
		# bert_model.predict(batch_string)

		# -------------------------------------------------------------------

		# # Batch Training - requires dataframe
		# # each index of each list must corresponds to the same pdf or contents
		# directory = []
		# file = []
		# text = [] # Extracted text from pdf
		# label = []
		# dataframe = pd.DataFrame(list(zip(directory, file, text, label)), columns=['directory', 'file', 'text', 'label'])

		# # Fit the training data into the model and call the training function
		# # if re-training, set a new directory before calling the train()
		# bert_model.fit(dataframe=dataframe)
		# # bert_model.set_output_model_directory(new_directory)
		# # bert_model.output_model_directory(new_directory)
		# bert_model.train()

		# # After training, save the label list in case of any newly added label(s)
		# # Makeshift label list saver
		# # Actual labels can be retrieved from bert_model.label_list
		# label_file = open("label.txt", "w")
		# label_list = bert_model.label_list
		# for label in label_list:
		# 	label_file.write(label + "\n")
		# label_file.close()

		return redirect('/upload' , labelsList=labelsList)


@app.route("/label")
def label():
    return render_template(
        "label.html",
        title="Label Documents", 
		labelsList=labelsList
		
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
	return render_template("view.html", title="View Documents", labelsList=labelsList)

@app.route("/pdf")
def pdf():
    return render_template(
            "pdf.html",
            title="render Documents",
    )