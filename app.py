from flask import Flask,render_template,request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app=Flask(__name__)

model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization',methods=["POST"])
def summarize():

    if request.method=="POST":

        inputtext=request.form["inputtext"]

        input_text="summarize:"+ inputtext

        tokenized_text=tokenizer.encode(input_text,return_tensor='pt', max_length=512).to(device)
        summary_=model.generate(tokenized_text, min_length=10, max_length=100)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

        '''
        text=<start> I am Vansh <end>
        vocab={<start>:1, I:2, am:3, Vansh:4, <end>:5}

        token ={I, am , Vansh}
        encode=[1,2,3]

        summary_=[[4,3,1,5]]
        summary=Vansh I
        '''
        
    return render_template('output.html',data={"summary":summary})

if __name__=='__main__':
    app.run()