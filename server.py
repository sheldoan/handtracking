# serve.py

from flask import Flask
from flask import render_template

from glob import glob
import json

# creates a Flask application, named app
app = Flask(__name__)


@app.route("/output/<path:path>")
def api_assets(path):
    return send_from_directory('output', path)

@app.route("/videos")
def api_videos():
    video_names = []
    for video_name in sorted(glob("static/" + "*.webm")):
        video_names.append(video_name)
    print("Video names length", len(video_names))
    return json.dumps(video_names)

@app.route("/")
def hello():
    message = "hello"
    return render_template('index.html', message=message)

# run the application
if __name__ == "__main__":
    app.run(debug=True)
