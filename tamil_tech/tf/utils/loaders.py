import os
import re
import yaml
import requests
import tensorflow as tf
from collections import UserDict
from tamil_tech.tf.models import Conformer 
from tamil_tech.tf.utils.tf_utils import preprocess_paths, append_default_keys_dict, check_key_in_dict
from tamil_tech.tf.utils.speech_featurizer import TFSpeechFeaturizer
from tamil_tech.tf.utils.text_featurizer import CharFeaturizer

DEFAULT_YAML = "tamil_tech/configs/conformer_s.yml"

def load_yaml(path):
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(preprocess_paths(path), "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)

class UserConfig(UserDict):
    """ User config class for training, testing or infering """

    def __init__(self, default: str, custom: str, learning: bool = True):
        assert default, "Default dict for config must be set"
        default = load_yaml(default)
        custom = append_default_keys_dict(default, load_yaml(custom))
        super(UserConfig, self).__init__(custom)
        if not learning and self.data.get("learning_config", None) is not None:
            # No need to have learning_config on Inferencer
            del self.data["learning_config"]
        elif learning:
            # Check keys
            check_key_in_dict(
                self.data["learning_config"],
                ["augmentations", "dataset_config", "optimizer_config", "running_config"]
            )
            check_key_in_dict(
                self.data["learning_config"]["dataset_config"],
                ["train_paths", "eval_paths", "test_paths"]
            )
            check_key_in_dict(
                self.data["learning_config"]["running_config"],
                ["batch_size", "num_epochs", "outdir", "log_interval_steps",
                    "save_interval_steps", "eval_interval_steps"]
            )

    def __missing__(self, key):
        return None

CONFORMER_L = UserConfig(DEFAULT_YAML, "tamil_tech/configs/conformer_l.yml", learning=True)
CONFORMER_M = UserConfig(DEFAULT_YAML, "tamil_tech/configs/conformer_m.yml", learning=True)
CONFORMER_S = UserConfig(DEFAULT_YAML, "tamil_tech/configs/conformer_s.yml", learning=True)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_conformer_model(model_path=None, tflite=True, greedy=True, conformer_type='L'):
    if conformer_type == 'L':
        state_size = 640
        blank = 0
        config = CONFORMER_L
    elif conformer_type == 'M':
        state_size = 640
        blank = 0
        config = CONFORMER_M
    else:
        state_size = 320
        blank = 0
        config = CONFORMER_S

    if model_path is None and tflite:
        model_path = 'tamil_conformer_tflite_' + str(greedy) + '_' + conformer_type + '.tflite'
    
    if model_path is None and not tflite:
        model_path = 'tamil_conformer_' + conformer_type + '.h5'
    
    if tflite and greedy:
        if os.path.exists(model_path):
            pass
        else:
            print("Downloading Model...")
            file_id = '1_6bmyWGwLSLB95I7_lKK7gILYAb7uYzR'
            download_file_from_google_drive(file_id, model_path)
            print("Downloaded Model Successfully...")
        model = tf.lite.Interpreter(model_path=model_path)
        return model, state_size, blank
    
    elif tflite and not greedy:
        if os.path.exists(model_path):
            pass
        else:
            print("Downloading Model...")
            file_id = '1rDez_Kt3tTh_8HETDrTedGsCAZYmERfu'
            download_file_from_google_drive(file_id, model_path)
            print("Downloaded Model Successfully...")
        model = tf.lite.Interpreter(model_path=model_path)
        return model, state_size, blank
    
    else:
        if os.path.exists(model_path):
            pass
        else:
            print("Downloading Model...")
            file_id = '1wSPhfjBjty3FKqEsbttPjZyjh2uwGLF6'
            download_file_from_google_drive(file_id, model_path)
            print("Downloaded Model Successfully...")
            
        speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
        text_featurizer = CharFeaturizer(config["decoder_config"])
        
        model = Conformer(
            vocabulary_size=text_featurizer.num_classes,
            **config["model_config"]
        )
        model._build(speech_featurizer.shape)
        model.load_weights(model_path, by_name=True)
        model.add_featurizers(speech_featurizer, text_featurizer)   

        return model, state_size, blank