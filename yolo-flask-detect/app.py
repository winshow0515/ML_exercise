#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(img, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            detections.append({
                'class': results.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        
        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Server starting with HTTPS")
    print("📱 Open on your phone: https://192.168.0.239:5000")
    print("⚠️  You will see a security warning - click 'Advanced' then 'Proceed'")
    print("="*60 + "\n")
    
    # Use adhoc SSL - Flask generates temp certificate automatically
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')