import os
from flask import Flask, request, render_template
import datetime
import internal_doc_chatbot
import pandas as pd

app = Flask(__name__, template_folder=os.path.join(path_root, 'internal-doc-chatbot'))
           
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        text_input = request.form['text_input']
        text_output, links = process_text(text_input)
        print(text_output)
        return render_template('index.html', text_output=text_output, links=links)
    return render_template('index.html')

def parse_numbers(s):
  return [float(x) for x in s.strip('[]').split(',')]

def return_Confluence_embeddings():
    # Today's date
    today = datetime.datetime.today()
    # Current file where the embeddings of our internal Confluence document is saved
    Confluence_embeddings_file = 'DOC_title_content_embeddings.csv'
    # Run the embeddings again if the file is more than a week old
    # Otherwise, read the save file
    Confluence_embeddings_file_date = datetime.datetime.fromtimestamp(os.path.getmtime(Confluence_embeddings_file))
    delta = today - Confluence_embeddings_file_date
    if delta.days > 7:
        DOC_title_content_embeddings= internal_doc_chatbot.update_internal_doc_embeddings()
    else:
        DOC_title_content_embeddings= pd.read_csv(Confluence_embeddings_file, dtype={'embeddings': object})
        DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings['embeddings'].apply(lambda x: parse_numbers(x))
 
    return DOC_title_content_embeddings
        
def process_text(query):
    
    DOC_title_content_embeddings= return_Confluence_embeddings()
    output, links = internal_doc_chatbot.internal_doc_chatbot_answer(query, DOC_title_content_embeddings)
    
    return output, links

if __name__ == '__main__':
    app.run()
