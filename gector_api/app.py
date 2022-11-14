from flask import Flask, request
from gector_api.predict import correct_sentences
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

@app.route('/correct')
def correct_sentence():
    working_dir = os.getcwd()
    request_json = request.get_json()
    sentence = request_json.get('sentence')

    corrected = correct_sentences(
        sentences=sentence,
        vocab_path=f'{working_dir}/data/output_vocabulary/',
        model_path=f'{working_dir}/models/xlnet_0_gectorv2.th',
        max_len=200,
        min_len=3,
        iteration_count=5,
        min_error_probability=0.0,
        lowercase_tokens=0,
        transformer_model='xlnet',
        special_tokens_fix=1,
        additional_confidence=0,
        batch_size=128,
        additional_del_confidence=0,
        is_ensemble=0,
        normalize='store_true',
        weights=None
    )
    return {
        'corrected':corrected
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
