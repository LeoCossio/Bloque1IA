#Python libraries
from flask import Flask, request, jsonify, render_template
import numpy as np
from load import joblib
import os
from werkzeug.utils import secure_filename

# load model
dt=joblib.load("dt.joblib")
# Create flask app
server = Flask(_name_)

#define a route to send JSON data
@server.route("/predictjson", methods=["POST"])