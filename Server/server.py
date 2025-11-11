from flask import Flask, request, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from kosenface import find_similar_faces

app = Flask(__name__, static_folder='static', static_url_path='/')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, supports_credentials=True, responses={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@socketio.on('filepath', namespace='/demo')
def emit_similar_faces(face_filepath):
    results = find_similar_faces(face_filepath)
    return emit('similar_faces', results)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, debug=True)

