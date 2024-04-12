import os
from mtcnn.mtcnn import MTCNN
import cv2


IMAGE_SIZE = (224, 224)
path = 'C:\\Users\\Lenovo\\Documents\\Github\\Datasets\\LFW\\lfw-deepfunneled\\lfw-deepfunneled'
new_path = 'C:\\Users\\Lenovo\\Documents\\Github\\Datasets\\LFW-edited'


# extract a single face from a given photograph
def extract_face(img_path, required_size=(160, 160)):
    # load image from file
    image = cv2.imread(img_path)
    
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(image)

    if len(results) < 1:
        return None

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1:y2, x1:x2]
    # resize pixels to the model size
    face = cv2.resize(face, required_size)
    return face


for person_name in os.listdir(path):
    
    for num, img_name in enumerate(os.listdir(os.path.join(path, person_name))):
        img_path = os.path.join(path, person_name, img_name)
        path_to_write = os.path.join(new_path, person_name)
        if os.path.isfile(os.path.join(path_to_write, img_name)):
            continue
        img = extract_face(img_path, required_size=IMAGE_SIZE)
        if img is None:
            print(f'----------- NO FACE IN {img_name} -----------')
            continue
        if not os.path.exists(path_to_write):
        # Create the directory if it does not exist
            os.makedirs(path_to_write)
        cv2.imwrite(os.path.join(path_to_write, img_name), img)
    print(f'{num+1} images saved in {path_to_write}')