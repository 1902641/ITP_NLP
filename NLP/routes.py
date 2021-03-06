from flask import render_template, url_for, flash, redirect, request, abort, send_from_directory, jsonify
from flask.sessions import NullSession
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
from NLP import app, pdfManagement, csvReader
import flask_excel as excel
import urllib.request, json
from flask import send_file
import os
from NLP.nlp_model.BERTModel import BERTModel
import pandas as pd
from datetime import date
import shutil

ALLOWED_EXTENSIONS = set(['pdf'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


categories_list = pdfManagement.retrieveCategories()

label_file = open("./NLP/nlp_model/label.txt", "r")
categories_list = label_file.read().splitlines()


@app.route("/")
def home():
    data = csvReader.readCSVPredict()
    return render_template("home.html", data = data, categories_list=categories_list)


@app.route('/upload', methods=['POST', 'GET'])
def upload_form():
    return render_template('upload.html', categories_list=categories_list)


@app.route('/train', methods=['POST'])
def upload_train():
    print("Request sent here")
    if request.method == 'POST':
        if request.files['files'].filename != '' and request.files['val_files'].filename != '' and request.files['labels'].filename != '':
            files = request.files.getlist('files')
            val_files = request.files.getlist('val_files')
            training_dataframe = pd.read_csv(request.files.get('labels'))
            training_dataframe = training_dataframe.iloc[:, 1:]
            file_name_list = []
            training_label = []
            text_extraction = []
            val_file_name_list = []
            val_training_label = []
            val_text_extract = []
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
            for val_file in val_files:
                if val_file and allowed_file(val_file.filename):
                    filename = secure_filename(val_file.filename)
                    val_file.save(os.path.join(app.config['TRAIN_FOLDER'], filename))
                    print(val_file.filename, ' file saved')
                    val_file_name_list.append(val_file.filename)
                    found_label = training_dataframe[
                        training_dataframe['filepath'].str.contains(val_file.filename, case=False, regex=False)]
                    val_training_label.append(found_label['ProductTypeCode'].iloc[0])
                    val_text_extract.append(pdfManagement.extract_pdf(val_file.filename))
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
            if len(check_set) == 0:
                retrain_flag = True
            else:
                for check_label in check_set:
                    if check_label not in label_list:
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
                    shutil.rmtree(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nlp_model', 'bert_model')),
                                  ignore_errors=True)
                    try:
                        os.mkdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nlp_model', 'bert_model')))
                    except OSError as e:
                        print("Error occurred when creating directory for model to save")
                        print(e.strerror)
                except OSError as e:
                    print("Error occurred when removing past model")
                    print(e.strerror)

            dataframe = pd.DataFrame(list(zip(file_name_list, text_extraction, training_label)),
                                     columns=['file', 'text', 'label'])

            val_dataframe = pd.DataFrame(list(zip(val_file_name_list, val_text_extract, val_training_label)),
                                                columns=['file', 'text', 'label'])
            # # Load the labels first before loading model
            bert_model = BERTModel()
            bert_model.fit(dataframe=dataframe, val_dataframe=val_dataframe, in_labels=label_list)
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

@app.route('/uploading', methods=['GET', 'POST'])
def upload_file():
    global filename
    file = None
    if request.method == "POST":
        if request.files:
            for key, f in request.files.items():
                if key.startswith('file'):
                    f.filename = secure_filename(f.filename)
                    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
                    flash('You were successfully logged in')
            # -------------------------------------------------------------------
            # # Assuming this is where text extraction and prediction will be done
            # # Makeshift label list loader
            label_file = open("./NLP/nlp_model/label.txt", "r")
            label_list = label_file.read().splitlines()

            # # Load the labels first before loading model
            bert_model = BERTModel()
            bert_model.load_label_list(label_list)
            predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
            file_names, text_extracted = pdfManagement.retrieveListOfPDF(predict_history_dataframe['file'].tolist())
            if len(file_names) < 1:
                return redirect('/upload')
            # Clean text for model to predict
            text_extracted = list(map(bert_model.clean_text, text_extracted))
            file_dataframe = pd.DataFrame(list(zip(file_names, text_extracted)), columns=["file", "text"])
            # Call load model if configuration are done
            bert_model.load_model()

            # Determine if single prediction or batch
            single_prediction = False
            print("Length of files for predict: ", len(file_names))
            if len(file_names) == 1:
                single_prediction = True
            # Map predicted results to each row in dataframe
            result_prob = []
            predict_label = []
            predict_label_code = []
            print(file_dataframe.head())
            if(single_prediction):
                predict_results = bert_model.predict(file_dataframe['text'].iloc[0], single_prediction=single_prediction)
                result_prob.append(predict_results[1])
                predict_label.append(predict_results[3])
                predict_label_code.append(predict_results[2])
            else:
                predict_results = bert_model.predict(file_dataframe['text'], single_prediction=single_prediction)
                for x in predict_results:
                    result_prob.append(x[1])
                    predict_label.append(x[3])
                    predict_label_code.append(x[2])
            file_dataframe['predict_label_code'] = predict_label_code
            file_dataframe['predict_label'] = predict_label
            file_dataframe['probabilities'] = result_prob
            file_dataframe['manual_check'] = 'False'
            upload_date = date.today().strftime("%d/%m/%Y")
            file_dataframe['upload_date'] = upload_date
            predict_history_dataframe = predict_history_dataframe.append(file_dataframe.filter(['file', 'predict_label', 'probabilities', 'predict_label_code', 'manual_check', 'upload_date'], axis=1))
            predict_history_dataframe.to_csv("./NLP/nlp_model/predict.csv", encoding='utf-8', index=False)
            # print(file_dataframe[['file', 'predicted_label']])
            return redirect(url_for('upload_form'))


@app.route("/label")
def label():
    return render_template("label.html", title="Label Documents",  
                           categories_list=categories_list)


@app.route("/export")
def export_csv():
    return send_file(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nlp_model', 'predict.csv')), as_attachment=True)


@app.route('/index_get_data')
def stuff():
    # Assume data comes from somewhere else
    file = request.args.get('file')
    predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
    raw_prob_list = predict_history_dataframe['probabilities'].tolist()
    prob_list = []
    predicted_label_list = predict_history_dataframe['predict_label_code'].tolist()
    confidence_list = []
    for prob_list_item in raw_prob_list:
        temp_list = prob_list_item.replace('[', '').replace(']','').replace(' ', '').split(',')
        prob_list.append(temp_list)
    for probability, label_code in zip(prob_list, predicted_label_list):
        confidence_list.append(probability[label_code].replace('\'', ''))
    data_list = [
        {
            "PDF_Name": file_name,
            "Label_Attached": predict_label,
            "Confidence_Level": confidence,
            "DateOfUpload": upload_date,
            "ManualCheck": f'{manual_check}'
        } for file_name, predict_label, confidence, upload_date, manual_check
        in zip(predict_history_dataframe['file'], predict_history_dataframe['predict_label'],
               confidence_list, predict_history_dataframe['upload_date'],
               predict_history_dataframe['manual_check'])]
    data = {
        "data": data_list
    }
    predic = predict_history_dataframe["probabilities"].tolist()
    row = 0
    for i,v in enumerate(data["data"]):
        if v['PDF_Name']==file:
            row=i
            break
    a_file = open("./NLP/nlp_model/label.txt")
    file_contents = a_file.read()
    label_headings = file_contents.splitlines()
    data2 = []
    probability_array = predic[row][1:-1].split(",")
    for i in range(len(label_headings)):
        temp_dict = {}
        temp_dict["PDF_Name"] = label_headings[i]
        temp_dict["Confidence_Level"] = probability_array[i]
        data2.append(temp_dict)

    
    data2 = {
        "data": data2
    }


    return jsonify(data2)


@app.route('/upload_get_data')
def uploadData():
    # Assume data comes from somewhere else
    predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
    raw_prob_list = predict_history_dataframe['probabilities'].tolist()
    prob_list = []
    predicted_label_list = predict_history_dataframe['predict_label_code'].tolist()
    confidence_list = []
    for prob_list_item in raw_prob_list:
        temp_list = prob_list_item.replace('[', '').replace(']','').replace(' ', '').split(',')
        prob_list.append(temp_list)
    for probability, label_code in zip(prob_list, predicted_label_list):
        confidence_list.append(probability[label_code].replace('\'', ''))
    data_list = [
        {
            "PDF_Name": file_name,
            "Label_Attached": predict_label,
            "Confidence_Level": confidence,
            "DateOfUpload": upload_date,
            "ManualCheck": f'{manual_check}',
            "VerifiedLabel": str(verified_label),
        } for file_name, predict_label, confidence, upload_date, manual_check, verified_label
        in zip(predict_history_dataframe['file'], predict_history_dataframe['predict_label'],
               confidence_list, predict_history_dataframe['upload_date'],
               predict_history_dataframe['manual_check'], predict_history_dataframe['verified_label'])]
    data = {
        "data": data_list
    }


    # data = {
    #     "data": [
    #         {
    #             "PDF_Name": "Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "80%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "pdf.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True" },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Search.pdf",
    #             "Label_Attached": "Search",
    #             "Confidence_Level": "67%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         },
    #         {
    #             "PDF_Name": "Fire Resistance Door.pdf",
    #             "Label_Attached": "Door",
    #             "Confidence_Level": "88%",
    #             "DateOfUpload": "6/6/2021",
    #             "ManualCheck": "True"
    #         }]
    # }
    return jsonify(data)


@app.route("/pdf")
def pdf():
    return render_template(
        "pdf.html", title="render Documents", categories_list=categories_list
    )

@app.route('/view', methods=['GET', 'POST'])
def verify():
        predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
        raw_prob_list = predict_history_dataframe['probabilities'].tolist()
        prob_list = []
        predicted_label_list = predict_history_dataframe['predict_label_code'].tolist()
        confidence_list = []
        for prob_list_item in raw_prob_list:
            temp_list = prob_list_item.replace('[', '').replace(']','').replace(' ', '').split(',')
            prob_list.append(temp_list)
        for probability, label_code in zip(prob_list, predicted_label_list):
            confidence_list.append(probability[label_code].replace('\'', ''))
        data_list = [
        {
            "PDF_Name": file_name,
            "Label_Attached": predict_label,
            "Confidence_Level": confidence,
            "DateOfUpload": upload_date,
            "ManualCheck": f'{manual_check}',
            "VerifiedLabel": str(verified_label),
        } for file_name, predict_label, confidence, upload_date, manual_check, verified_label
        in zip(predict_history_dataframe['file'], predict_history_dataframe['predict_label'],
                confidence_list, predict_history_dataframe['upload_date'],
                predict_history_dataframe['manual_check'], predict_history_dataframe['verified_label'])]
        count = 0
        unchecked_count = 0
        label_attached = ""
        confidence_level=0
        cl = []
        for i in data_list:
            count +=1
            if i["VerifiedLabel"] == "nan":
                file = i["PDF_Name"]
                unchecked_count+=1
                label_attached = i["Label_Attached"]
                x = i["Confidence_Level"]
                
                cl = x.replace('[', '').replace(']','').replace('\'','').replace(' ', '').replace('%', '').split(',')
                [float(i) for i in cl]
        percentage = (count-unchecked_count)/count*100
        if cl != []:
            confidence_level = max(cl)
        else:
            return render_template('upload.html')
        return render_template('view.html', file=file, percentage=percentage, label_attached=label_attached, confidence_level=confidence_level)



@app.route("/trained_stats")
def trained_stats():
    a_file = open("./NLP/nlp_model/label.txt")
    file_contents = a_file.read()
    label_headings = file_contents.splitlines()
    predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
    raw_prob_list = predict_history_dataframe['probabilities'].tolist()
    prob_list = []
    predicted_label_list = predict_history_dataframe['predict_label_code'].tolist()
    confidence_list = []
    for prob_list_item in raw_prob_list:
        temp_list = prob_list_item.replace('[', '').replace(']','').replace(' ', '').split(',')
        prob_list.append(temp_list)
    for probability, label_code in zip(prob_list, predicted_label_list):
        confidence_list.append(probability[label_code].replace('\'', ''))
    data_list = [
        {
            "PDF_Name": file_name,
            "Label_Attached": predict_label,
            "Confidence_Level": confidence,
            "DateOfUpload": upload_date,
            "ManualCheck": f'{manual_check}'
        } for file_name, predict_label, confidence, upload_date, manual_check
        in zip(predict_history_dataframe['file'], predict_history_dataframe['predict_label'],
               confidence_list, predict_history_dataframe['upload_date'],
               predict_history_dataframe['manual_check'])]

    headings_count = {}
    for i in data_list:
        if i["Label_Attached"] in headings_count:
            headings_count[i["Label_Attached"]] += 1
        else:
            headings_count[i["Label_Attached"]] = 1
    
    data = []
    for key in headings_count:
        data.append({ "name":key,"y":headings_count[key]})

    return jsonify(data)

@app.route("/verified_stats")
def verified_stats():
    a_file = open("./NLP/nlp_model/label.txt")
    file_contents = a_file.read()
    label_headings = file_contents.splitlines()
    predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
    raw_prob_list = predict_history_dataframe['probabilities'].tolist()
    prob_list = []
    predicted_label_list = predict_history_dataframe['predict_label_code'].tolist()
    confidence_list = []
    for prob_list_item in raw_prob_list:
        temp_list = prob_list_item.replace('[', '').replace(']','').replace(' ', '').split(',')
        prob_list.append(temp_list)
    for probability, label_code in zip(prob_list, predicted_label_list):
        confidence_list.append(probability[label_code].replace('\'', ''))
    data_list = [
        {
            "PDF_Name": file_name,
            "Label_Attached": predict_label,
            "Confidence_Level": confidence,
            "DateOfUpload": upload_date,
            "ManualCheck": f'{manual_check}',
            "VerifiedLabel": str(verified_label),
        } for file_name, predict_label, confidence, upload_date, manual_check, verified_label
        in zip(predict_history_dataframe['file'], predict_history_dataframe['predict_label'],
               confidence_list, predict_history_dataframe['upload_date'],
               predict_history_dataframe['manual_check'], predict_history_dataframe['verified_label'])]
    headings_count = {}
    for i in data_list:
        if i["VerifiedLabel"] in headings_count:
            headings_count[i["VerifiedLabel"]] += 1
        else:
            headings_count[i["VerifiedLabel"]] = 1
    
    data = []
    for key in headings_count:
        data.append({ "name":key,"y":headings_count[key]})

    return jsonify(data)

@app.route("/accuracy_stats")
def accuracy_stats():
    a_file = open("./NLP/nlp_model/label.txt")
    file_contents = a_file.read()
    label_headings = file_contents.splitlines()
    predict_history_dataframe = pd.read_csv("./NLP/nlp_model/predict.csv")
    raw_prob_list = predict_history_dataframe['probabilities'].tolist()
    prob_list = []
    predicted_label_list = predict_history_dataframe['predict_label_code'].tolist()
    confidence_list = []
    for prob_list_item in raw_prob_list:
        temp_list = prob_list_item.replace('[', '').replace(']','').replace(' ', '').split(',')
        prob_list.append(temp_list)
    for probability, label_code in zip(prob_list, predicted_label_list):
        confidence_list.append(probability[label_code].replace('\'', ''))
    data_list = [
        {
            "PDF_Name": file_name,
            "Label_Attached": predict_label,
            "Confidence_Level": confidence,
            "DateOfUpload": upload_date,
            "ManualCheck": f'{manual_check}',
            "VerifiedLabel": str(verified_label),
        } for file_name, predict_label, confidence, upload_date, manual_check, verified_label
        in zip(predict_history_dataframe['file'], predict_history_dataframe['predict_label'],
               confidence_list, predict_history_dataframe['upload_date'],
               predict_history_dataframe['manual_check'], predict_history_dataframe['verified_label'])]
    headings_count = {}
    total_verified_docs = 0
    total_matched_docs = 0
    for i in data_list:
        if i["VerifiedLabel"] != "nan":
            total_verified_docs += 1
            if i["VerifiedLabel"] == i["Label_Attached"]:
                total_matched_docs += 1

    
    data = []
    data = { "matched":total_matched_docs,"total": total_verified_docs}

    return jsonify(data)

