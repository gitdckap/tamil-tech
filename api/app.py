import tensorflow as tf
from tensorflow_asr.utils import setup_environment, setup_devices

setup_environment()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})

setup_devices([0], cpu=False)

import numpy as np
from flask import Flask, request, render_template
from tamil_tech.zero_shot import ConformerTamilASR

app = Flask(__name__, template_folder="template")
speech_recognizer = ConformerTamilASR(path='ConformerS_new.h5')

@app.route("/", methods=["GET", "POST"])
def predict():
	""" 
		Returns tamil transcription data of the uploaded audio data
	"""
	if request.method == "POST":
		try:
			# read uplaoded audio file as bytes
			file = request.files['file'].read()
			# get the audio signal as numpy array
			signal = speech_recognizer.read_raw_audio(file)
			# convert audio isgnal to spectrogram using speech featurizer and expand tensor at 0th dimension
			signal = speech_recognizer.model.speech_featurizer.tf_extract(tf.convert_to_tensor(signal, dtype=tf.float32))
			signal = tf.expand_dims(signal, axis=0)
			# recognize for text sequence using greedy decoding
			pred = speech_recognizer.model.recognize(features=signal)
			# convert sequence to tamil unicode data
			pred = speech_recognizer.bytes_to_string(pred.numpy())[0]
			# retun the prediction to the user
			return pred
        except Exception as e:
			print(e)
			pass
	else:
		return "Error 404. Page Not Found!"

if __name__=="__main__":
	app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)