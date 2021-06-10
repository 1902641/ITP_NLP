import csv
import os

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