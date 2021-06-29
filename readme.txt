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
Extract the model.ckpt-9.zip.001 in the same directory ("NLP/nlp_model/bert_model") using an archive application (e.g., 7zip)

Label Page : 
    - Retrieve all images and display (With page number)
        -
    - Display list of PDF names (Retrieve all existing pdf in file and display name)
        - Clickable list of PDF names to call the relevant image for pdf disp941\
        lay (Search feature KIV)
    - System Generates List of Labels (Generate list of labels on Select PDF)
    - Upload new PDF files - Somewhat done but now got error again fml

