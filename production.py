
import asyncio
from asyncore import loop
from datetime import datetime
from flask import jsonify, make_response
from flask import Flask, request, send_file
# from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
import os
import cv2
import base64

from FingerprintImageEnhancer import FingerprintImageEnhancer
import numpy as np

async def processImage(image):
        segmentation_mask = cv2.imread(image)
        ycbcr_image = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2YCrCb)      
        ycbcr_image = ycbcr_image[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(120, 120))
        clahe_image = clahe.apply(ycbcr_image)

        image_enhancer = FingerprintImageEnhancer() 
        out = image_enhancer.enhance(clahe_image) 
        out = out * 255

        out = out.astype('uint8') 
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(out)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        min_size = 50

            #your answer image
        finalImage = np.ones((output.shape))*255
            #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                finalImage[output == i + 1] = 0

        cv2.flip(finalImage,1,finalImage)
        cv2.imwrite('finger.jpg', finalImage)
        
loop = asyncio.get_event_loop()

app = Flask(__name__)
cors = CORS(app)
path = os.getcwd()
UPLOAD_PATH = os.path.join(path, 'uploads')

ALLOWED_FILE = set(['png', 'jpg', 'jpeg'])

def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE


app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config['FILE_SIZE'] = 16 * 1024 * 1024

@app.route('/api/v1/processfinger', methods=['POST'])
def verifyUser():
    if request.method == 'POST':
        files = request.files.values()
        for index, file in enumerate(files):
            if not os.path.isdir(UPLOAD_PATH):
                os.makedirs(UPLOAD_PATH)
            nadraPic = os.path.join(
            UPLOAD_PATH, "finger.jpg")
            file.save(nadraPic)
        loop.run_until_complete(processImage(nadraPic))
        with open("finger.jpg", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            return jsonify({'fingerprint': 
                            str(b64_string)})
    
        # send_file('finger.jpg',
        #             mimetype="image/jpeg",
        #             as_attachment= True)

        
                


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000,  threaded=True)