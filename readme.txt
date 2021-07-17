## Quick Start

```bash
# Clone the repository
git clone https://github.com/1902641/ITP_NLP

# Go inside the directory
cd ITP_NLP

# Checkout nlp-model-cleanup branch
git checkout nlp-model-cleanup 

# Get python 3.7 64 bits and set path in windows 10
https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe

after installation, you have to make sure path is set before Microsoft store path. (might need more steps if you have multiple python versions like 2.7)

# create venv (first time only) 
python -m venv ./venv

# activate venv (do this everytime you are working on the project) run `deactivate` when done
venv/Scripts/activate

# Install dependencies
python -m pip install -r requirements.txt

# Extract model
open file explorer and navigate to ITP_NLP/NLP/nlp_model/bert_model, double click on model.ckpt-19.zip.001 and then extract.

#start website
python run.py

#default port is 5000
browse localhost:5000 in browser

```




venv/Scripts/activate

pip install the following but it should be in the venv
flask
tox - #Used to run the test suite against multiple Python versions.
jinja2
pdf2image
fpdf
conda 
poppler
PyPDF2
Flask-Dropzone
wtforms
Flask-Excel
tablib[pandas]

The model is saved and restored using checkpoints. Due to size limitations,
the check point data file is compressed and split into parts in the following directory:
"NLP/nlp_model/bert_model"
Extract the model.ckpt-19.zip.001 in the same directory ("NLP/nlp_model/bert_model") using an archive application (e.g., 7zip)

Label Page : 
    - Retrieve all images and display (With page number)
        -
    - Display list of PDF names (Retrieve all existing pdf in file and display name)
        - Clickable list of PDF names to call the relevant image for pdf disp941\
        lay (Search feature KIV)
    - System Generates List of Labels (Generate list of labels on Select PDF)
    - Upload new PDF files - Somewhat done but now got error again fml

