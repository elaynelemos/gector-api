from flask import Flask, request
from gector_api.predict import correct_sentences

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

@app.route('/correct', methods=['POST'])
def correct_sentence():
    request_json = request.get_json()
    sentence = request_json.get('sentence')

    corrected = correct_sentences(
        sentences=sentence,
        batch_size=128,
        normalize=None,
    )
    return {
        'corrected':corrected
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
