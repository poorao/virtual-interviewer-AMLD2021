import random
import time
import logging

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from flask import Flask, request
from flask_cors import CORS
from interact import follow_up_generator


app = Flask(__name__)
app.config["SECRET_KEY"] = "via"
CORS(app)

his_file = open("history.txt", "a")

# file which has all the pre-defined questions
questionBank = "soq2.txt"
ques = open(questionBank, "r")
# SoQ (Script of Questions): Contains all the base questions asked in the interview
soq_all = ques.readlines()
soq = soq_all

args = {
    "model_checkpoint": "runs/Apr25_11-25-52_2367d6c33fcc_gpt2",
    "seed": 42,
    "device": "cuda",
    "no_sample": False,
    "max_length": 20, 
    "min_length": 1,
    "temperature": 0.7, 
    "top_k": 20, 
    "top_p": 0
}

question_status = 0
follow_up_status = -1

MAX_FOLLOW_UPS = 1
TOTAL_QUESTIONS = 4

history = list()


def reset_state():
    question_status = 0
    follow_up_status = -1
    return


def create_model(args):
    """ Load the fine-tuned model from the runs directory. Make sure to change the `model_checkpoint` to the correct trained directory """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    if args["model_checkpoint"] == "":
        raise ValueError(
            "Interacting with GPT2 requires passing a finetuned model_checkpoint"
        )

    random.seed(args["seed"])
    torch.random.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args["model_checkpoint"])
    model_class = GPT2LMHeadModel
    model = model_class.from_pretrained(args["model_checkpoint"])

    model.to(args["device"])
    model.eval()
    return (model, tokenizer)

model, tokenizer = create_model(args["model_checkpoint"])


@app.route("/")
def follow_up():
    global history
    global question_status
    global follow_up_status
    global soq
    pre_history = ""

    input_answer = request.args.get("user-response")
    print("INPUT:", input_answer)

    if input_answer == None:
        return "Sorry! I didn't get you."
    history.append(input_answer)
    fq = ""

    if question_status == -1:
        fq = "The interview is over. Thank you."
        question_status = 0
        follow_up_status = -1
        return fq
        # return ([],-1,0,fq)

    if question_status == 0 and follow_up_status == -1:  # First Question
        fq += "Good to meet you! Let's get started. " + soq[question_status]
        follow_up_status += 1

    elif follow_up_status >= MAX_FOLLOW_UPS:
        follow_up_status = 0
        question_status += 1
        if question_status >= TOTAL_QUESTIONS:
            fq = "That's it! Thank you! It was lovely talking to you."
            question_status = -1
        else:
            fq = soq[question_status]
            if question_status % 2 != 0:
                # time.sleep(6)
                print("wait")

    else:
        # Call the fine-tuned GPT-2 model followQG to generate follow-up question
        print("HISTORY:", history)
        pre_history = history[0]
        follow_up_status += 1
        fq = follow_up_generator(history, model, tokenizer, args)
        # time.sleep(5)

    if follow_up_status != 0 or question_status != 0:
        value = str(pre_history) + "," + str(input_answer) + "," + str(fq) + "\n"
        his_file.write(value)

    # re-initializing history as our model is trained only on one level of QA
    history = list()
    history.append(fq)
    print("Follow-up Question:", fq)

    return fq


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

