# import cv2
# import os
# import numpy as np
# from flask import Flask, render_template, Response, request, send_file
# from werkzeug.utils import secure_filename
# from flask import Flask, render_template, request, redirect, send_file
# import os
# from video_processing import process_video
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# # Function to detect lanes (dummy implementation)
# def detect_lanes(frame):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian Blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Use Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)
    
#     # Define a region of interest for lane detection (this is customizable)
#     height, width = edges.shape
#     mask = np.zeros_like(edges)
#     polygon = np.array([[(0, height), (width // 2, height // 2), (width, height)]], np.int32)
#     cv2.fillPoly(mask, polygon, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)
    
#     # Hough Line Transform to detect lines
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
    
#     # Draw the lines on the original frame
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
#     return frame

# # Camera stream route
# @app.route('/start-camera')
# def start_camera():
#     def generate():
#         cap = cv2.VideoCapture(0)  # Open the default camera
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Process the frame for lane detection
#             processed_frame = detect_lanes(frame)
            
#             # Encode frame as JPEG
#             ret, jpeg = cv2.imencode('.jpg', processed_frame)
#             if ret:
#                 frame = jpeg.tobytes()
#                 # Send the frame back as multipart HTTP response
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
#         cap.release()

#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Main page route
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process-video', methods=['POST'])
# def process_video_route():
#     if 'video' not in request.files:
#         return redirect('/')
#     video = request.files['video']
#     if video.filename == '':
#         return redirect('/')
#     input_path = os.path.join(UPLOAD_FOLDER, video.filename)
#     output_path = os.path.join(PROCESSED_FOLDER, f"processed_{video.filename}")
#     video.save(input_path)
#     process_video(input_path, output_path)
#     return render_template('download.html', video_path=output_path)

# @app.route('/download/<path:filename>')
# def download(filename):
#     return send_file(filename, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)


import cv2
import os
import numpy as np
from flask import Flask, render_template, Response, request, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from video_processing import process_video

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Function to detect lanes (dummy implementation)
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width // 2, height // 2), (width, height)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame

# Camera stream route
@app.route('/start-camera')
def start_camera():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Camera not accessible."

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detect_lanes(frame)
            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main page route
@app.route('/')
def index():
    return render_template('index.html')

# Video processing route
@app.route('/process-video', methods=['POST'])
def process_video_route():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No file selected."}), 400

    input_path = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{secure_filename(video.filename)}")

    video.save(input_path)
    try:
        process_video(input_path, output_path)
    except Exception as e:
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

    #return jsonify({"message": "Video processed successfully.", "download_url": f"/download/{output_path}"})
    return render_template('download.html', video_path=output_path)

# Download route
@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)