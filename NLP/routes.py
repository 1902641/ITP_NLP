from flask import render_template, url_for, flash, redirect, request, abort, send_from_directory, jsonify
from flask.sessions import NullSession
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


categories_list = pdfManagement.retrieveCategories()

labelsList = csvReader.readCSV()


@app.route("/")
def home():
    return render_template("home.html", labelsList=labelsList, categories_list=categories_list)


@app.route('/upload', methods=['POST', 'GET'])
def upload_form():
    return render_template('upload.html', categories_list=categories_list)


@app.route('/train', methods=['POST'])
def upload_train():
    print("Request sent here")
    if request.method == 'POST':
        if request.files['files'].filename != '' and request.files['labels'].filename != '':
            files = request.files.getlist('files')
            training_dataframe = pd.read_csv(request.files.get('labels'))
            training_dataframe = training_dataframe.iloc[:, 1:]
            file_name_list = []
            training_label = []
            text_extraction = []
            print("files uploaded")
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['TRAIN_FOLDER'], filename))
                    print(file.filename, ' file saved')
                    file_name_list.append(file.filename)
                    found_label = training_dataframe[
                        training_dataframe['filepath'].str.contains(file.filename, case=False, regex=False)]
                    training_label.append(found_label['ProductTypeCode'].iloc[0])
                    text_extraction.append(pdfManagement.extract_pdf(file.filename))
                else:
                    print('file saved failed')
            label_file = open("./NLP/nlp_model/label.txt", "r")
            label_list = label_file.read().splitlines()

            # Load list of documents that were used to train
            train_history_dataframe = pd.read_csv("./NLP/nlp_model/train.csv")
            new_train_history_dataframe = pd.DataFrame(list(zip(file_name_list, training_label)),
                                                       columns=['file', 'label'])

            # Check if there is a new label to train
            # If there is new label, model need to be retrain from scratch
            check_set = set(training_label)
            retrain_flag = False
            for check_label in check_set:
                if check_label in label_list:
                    retrain_flag = True
                    break
            print("Retrain: ", retrain_flag)

            if retrain_flag:
                old_name_list = [x for x in train_history_dataframe['file']]
                old_label_list = [y for y in train_history_dataframe['label']]
                old_text_extraction = [pdfManagement.extract_pdf(z) for z in train_history_dataframe['file']]
                file_name_list.extend(old_name_list)
                training_label.extend(old_label_list)
                text_extraction.extend(old_text_extraction)
                try:
                    os.rmdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nlp_model', 'bert_model')))
                    try:
                        os.mkdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nlp_model', 'bert_model')))
                    except OSError as e:
                        print("Error occurred when creating directory for model to save")
                except OSError as e:
                    print("Error occurred when removing past model")

            dataframe = pd.DataFrame(list(zip(file_name_list, text_extraction, training_label)),
                                     columns=['file', 'text', 'label'])

            # # Load the labels first before loading model
            bert_model = BERTModel()
            bert_model.fit(dataframe=dataframe, in_labels=label_list)
            bert_model.train()

            label_file = open("./NLP/nlp_model/label.txt", "w")
            label_list = bert_model.label_list
            for in_label in label_list:
                label_file.write(in_label + "\n")
            label_file.close()

            # Append current to past history and save overall history
            train_history_dataframe = train_history_dataframe.append(new_train_history_dataframe, ignore_index=True)
            train_history_dataframe.to_csv("./NLP/nlp_model/train.csv", encoding='utf-8', index=False)

            return render_template("train.html", train_flag=True)

    return render_template("train.html", train_flag=False)


# @app.route('/uploading', methods=['POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the files part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         files = request.files.getlist('files[]')
#         for file in files:
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         flash('File(s) successfully uploaded')
#         # -------------------------------------------------------------------
#         # # Assuming this is where text extraction and prediction will be done
#         # # Makeshift label list loader
#         label_file = open("./NLP/nlp_model/label.txt", "r")
#         label_list = label_file.read().splitlines()
#
#         # # Load the labels first before loading model
#         bert_model = BERTModel()
#         bert_model.load_label_list(label_list)
#
#         file_names, text_extracted = pdfManagement.retrieveListOfPDF()
#         # Clean text for model to predict
#         text_extracted = list(map(bert_model.clean_text, text_extracted))
#         file_dataframe = pd.DataFrame(list(zip(file_names, text_extracted)), columns=["file", "text"])
#         # # Optional - load or save to different directory
#         # # If loading different models, call this before load_model()
#         # new_directory = "./bert_model_v2"
#         # bert_model.set_output_model_directory(new_directory)
#         # bert_model.output_model_directory(new_directory)
#
#         # Call load model if configuration are done
#         bert_model.load_model()
#
#         # Determine if single prediction or batch
#         single_prediction = False
#         if len(file_names) == 1:
#             single_prediction = True
#         # Map predicted results to each row in dataframe
#         predict_results = bert_model.predict(file_dataframe['text'], single_prediction=single_prediction)
#         result_prob = []
#         predict_label = []
#         predict_label_code = []
#         for x in predict_results:
#             result_prob.append(x[1])
#             predict_label.append(x[3])
#             predict_label_code.append(x[2])
#         file_dataframe['predicted_label_code'] = predict_label_code
#         file_dataframe['predicted_label'] = predict_label
#         file_dataframe['probabilities'] = result_prob
#         print(file_dataframe[['file', 'predicted_label']])
#
#         return redirect('/upload', labelsList=labelsList)


@app.route('/uploading', methods=['GET', 'POST'])
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

            # -------------------------------------------------------------------
            # # Assuming this is where text extraction and prediction will be done
            # # Makeshift label list loader
            label_file = open("./NLP/nlp_model/label.txt", "r")
            label_list = label_file.read().splitlines()

            # # Load the labels first before loading model
            bert_model = BERTModel()
            bert_model.load_label_list(label_list)

            file_names, text_extracted = pdfManagement.retrieveListOfPDF()
            # Clean text for model to predict
            text_extracted = list(map(bert_model.clean_text, text_extracted))
            file_dataframe = pd.DataFrame(list(zip(file_names, text_extracted)), columns=["file", "text"])
            # # Optional - load or save to different directory
            # # If loading different models, call this before load_model()
            # new_directory = "./bert_model_v2"
            # bert_model.set_output_model_directory(new_directory)
            # bert_model.output_model_directory(new_directory)

            # Call load model if configuration are done
            bert_model.load_model()

            # Determine if single prediction or batch
            single_prediction = False
            if len(file_names) == 1:
                single_prediction = True
            # Map predicted results to each row in dataframe
            predict_results = bert_model.predict(file_dataframe['text'], single_prediction=single_prediction)
            result_prob = []
            predict_label = []
            predict_label_code = []
            for x in predict_results:
                result_prob.append(x[1])
                predict_label.append(x[3])
                predict_label_code.append(x[2])
            file_dataframe['predicted_label_code'] = predict_label_code
            file_dataframe['predicted_label'] = predict_label
            file_dataframe['probabilities'] = result_prob
            print(file_dataframe[['file', 'predicted_label']])
            return redirect("upload.html")


@app.route("/label")
def label():
    return render_template("label.html", title="Label Documents", labelsList=labelsList,
                           categories_list=categories_list)


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
                "PDF_Name": "Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "80%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "pdf.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
                "Label_Attached": "Door",
                "Confidence_Level": "88%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Search.pdf",
                "Label_Attached": "Search",
                "Confidence_Level": "67%",
                "DateOfUpload": "6/6/2021",
                "ManualCheck": "True"
            },
            {
                "PDF_Name": "Fire Resistance Door.pdf",
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
    return render_template("view.html", title="View Documents", labelsList=labelsList, file=file,
                           categories_list=categories_list)


@app.route("/pdf")
def pdf():
    return render_template(
        "pdf.html", title="render Documents", categories_list=categories_list
    )
