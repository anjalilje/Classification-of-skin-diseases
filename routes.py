from flask import Flask, render_template, request, send_file, send_from_directory
from scripts.upload import show_file, handle_file#, data_handler

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    return handle_file(request, app.config['UPLOAD_FOLDER'])

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
