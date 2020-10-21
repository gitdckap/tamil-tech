import os

TAMIL = os.path.abspath(os.path.join(os.path.dirname(__file__), "tamil_tech/vocabularies/new_tamil_vocab.txt"))
DEFAULT_YAML = "tamil_tech/configs/conformer_m.yml"

from tamil_tech.tf.utils.tf_utils import *
from tamil_tech.tf.utils.speech_featurizer import *
from tamil_tech.tf.utils.text_featurizer import *
# from tamil_tech.tf.utils.gammatone import *
# from tamil_tech.tf.utils.loaders import *