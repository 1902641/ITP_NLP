from flask import render_template, url_for, flash, redirect, request, abort, send_from_directory, jsonify
from flask.sessions import NullSession
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
from NLP import app, pdfManagement, csvReader
import flask_excel as excel
import urllib.request, json
import os


categories_list = pdfManagement.retrieveCategories()

labelsList = csvReader.readCSV()


@app.route("/")
def home():
 return render_template("home.html", labelsList=labelsList, categories_list = categories_list)


@app.route('/upload', methods=['POST','GET'])
def upload_form():
 return render_template('upload.html', categories_list = categories_list)


@app.route('/uploading', methods=['GET','POST'])
def upload_file():
    global filename
    file = None
    if request.method == "POST":
        if request.files:
            file = request.files["file"]
            if file.filename == "":
                print("No filename")
                return redirect("upload.html")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('You were successfully logged in')
            return redirect("upload.html")


@app.route("/label")
def label():
    return render_template( "label.html", title="Label Documents", labelsList=labelsList, categories_list = categories_list )


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
                "PDF_Name" : "pdf.pdf",
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
    file = request.args.get('file')
    return render_template("view.html", title="View Documents", labelsList=labelsList, file=file , categories_list = categories_list)

@app.route("/pdf")
def pdf():
    return render_template(
            "pdf.html", title="render Documents",categories_list = categories_list
    )

