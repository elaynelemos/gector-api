from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

@app.route('/correct')
def correct_sentence():
    return request.get_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
