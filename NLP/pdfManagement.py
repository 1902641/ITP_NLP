import os
import PyPDF2
from os import walk
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

mypath = os.path.join(os.path.dirname( __file__ ), 'static', 'uploads')
txtpath = os.path.join(os.path.dirname( __file__ ), 'static', 'text_extraction')
train_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'static', 'trains'))


def extract_pdf(filename) -> str:
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
    pdf_file = open(os.path.join(train_path, filename), 'rb')
    password = b""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(pdf_file, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=False):
        interpreter.process_page(page)

    text = retstr.getvalue()

    pdf_file.close()
    device.close()
    retstr.close()
    return text

def retrieveListOfPDF():
    f = []
    text_extract = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    for item in f:
        text_extract.append(pdfToTxt(item))
    return f, text_extract


def pdfToTxt(filename):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
    pdf_file = open(os.path.join(mypath, filename), 'rb')
    password = b""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(pdf_file, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=False):
        interpreter.process_page(page)

    text = retstr.getvalue()

    pdf_file.close()
    device.close()
    retstr.close()
    return text
    # #Open file Path
    # filename = filename.replace(' ', '_')
    # pdf_File = open(os.path.join(mypath, filename), 'rb')
    #
    # #Create PDF Reader Object
    # pdf_Reader = PyPDF2.PdfFileReader(pdf_File)
    # count = pdf_Reader.numPages # counts number of pages in pdf
    # TextList = []
    #
    # #Extracting text data from each page of the pdf file
    # for i in range(count):
    #     try:
    #         page = pdf_Reader.getPage(i)
    #         TextList.append(page.extractText())
    #     except:
    #         pass
    #
    # #Converting multiline text to single line text
    # TextString = " ".join(TextList)
    #
    # #Save text from pdf to pdf file
    # x = filename.split(".")
    # text_file = open(os.path.join(txtpath, x[0]+'.txt'), "w", encoding='utf-8')
    # text_file.write(TextString)
    # text_file.close()
    # return TextString
    
    