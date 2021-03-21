from flask import Flask, render_template, request
from wtforms import (Form, TextField, validators, SubmitField,DecimalField, IntegerField)
import json
app = Flask(__name__)

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    data_name = "PETR4_SA_1"
    # look_back = 15
    # epochs_num = 25
    # history = json.load(open("./../Models/{}results.json".format(data_name, data_name, epochs_num), 'r'))

    f = open("./../Models/{}/results.txt".format(data_name), "r")
    print(f.read())
    return (f.read())
    # return "<h1>Not Much Going On Here</h1>"


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # Run app
    app.run(host="0.0.0.0", port=80)

