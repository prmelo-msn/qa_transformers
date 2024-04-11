import os
from flask import Flask, render_template
from flask import request 
import json 
import transformers
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from transformers import pipeline


app = Flask(__name__)   

pathModel  = 'qa_diff'
#pathModel  = 'D:\\projetos\\qa_diff'
tokenizer = AutoTokenizer.from_pretrained(pathModel)
model     = TFAutoModelForQuestionAnswering.from_pretrained(pathModel)
nlp       = pipeline("question-answering", model=model,tokenizer=tokenizer)

def processQuestion(question,context):
    result = nlp(question=question, context=context)   
    return result['answer']

@app.route("/")
def hello():
 return "Hello World!"


@app.route("/index")
def index():
 return render_template('index.html')

@app.route("/question", methods = ['POST'])
def question():
    sQuestion = request.form['txtPergunta']
    sContext  = request.form['txtContexto']
    sAnswer   = processQuestion(sQuestion, sContext)

    result = {
        'sContext': sContext,
        'sQuestion': sQuestion,
        'sAnswer': sAnswer
    }

    return render_template('index.html', result=result)




if __name__ == "__main__":
 app.run(debug = True)