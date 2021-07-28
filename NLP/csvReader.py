import csv
import os
import pandas as pd

mypath = os.path.join(os.path.dirname( __file__ ), 'static', 'labels.csv')

def readCSV():
	x = []
	with open(mypath) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				x.append(row[1])
				line_count += 1
		x = list(dict.fromkeys(x))
		return x

	
def readCSVPredict():
	training_data = pd.read_csv("./NLP/nlp_model/train.csv")
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
	total_verified_docs = 0
	total_matched_docs = 0
	count = 0
	checked_count = 0
	for i in data_list:
		count +=1
		if i["VerifiedLabel"] != "nan":
			total_verified_docs += 1
			if i["VerifiedLabel"] == i["Label_Attached"]:
				total_matched_docs += 1
		if i["ManualCheck"] == "True":
			checked_count += 1
	accuracy = total_matched_docs/total_verified_docs
	x = readCSV()
	totalLabels = len(x)
	no_of_training_data = count
	total_manual_check_data = len(predict_history_dataframe)
	files_unchecked = total_manual_check_data - checked_count
	
	data={
		"no_of_training_data": str(no_of_training_data),
		"accuracy":str(accuracy)[0:5],
		"totalLabels":str(totalLabels),
		"total_manual_check_data":str(total_manual_check_data),
		"checked_count":str(checked_count),
		"files_unchecked":str(files_unchecked)
	}
	return data
