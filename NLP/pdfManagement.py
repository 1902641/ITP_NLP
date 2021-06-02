import os
import PyPDF2
from os import walk
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


mypath = os.path.join(os.path.dirname( __file__ ), 'static', 'uploads')
txtpath = os.path.join(os.path.dirname( __file__ ), 'static', 'text_extraction')

def retrieveListOfPDF():
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    for item in f:
        pdfToTxt(item)

def pdfToTxt(filename):
    #Open file Path
    pdf_File = open(os.path.join(mypath, filename), 'rb') 

    #Create PDF Reader Object
    pdf_Reader = PyPDF2.PdfFileReader(pdf_File)
    count = pdf_Reader.numPages # counts number of pages in pdf
    TextList = []

    #Extracting text data from each page of the pdf file
    for i in range(count):
        try:
            page = pdf_Reader.getPage(i)
            TextList.append(page.extractText())
        except:
            pass

    #Converting multiline text to single line text
    TextString = " ".join(TextList)
    
    #Save text from pdf to pdf file
    x = filename.split(".")
    text_file = open(os.path.join(txtpath, x[0]+'.txt'), "w", encoding='utf-8')
    text_file.write(TextString)
    text_file.close()
    
    