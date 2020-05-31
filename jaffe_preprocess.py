from parameters import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=SCALEFACTOR,
        minNeighbors=5
    )

    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        image = cv2.resize(image, (48, 48),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite('image' + '.png', image)
        image = image / 255.

    except Exception:
        print("[+] Problem during resize")
        return None
    return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d


def flip_image(image):
    return cv2.flip(image, 1)


def data_to_image(data, i):
    data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((256, 256))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()

    # if you want to save all images
    # cv2.imwrite(SAVE_DIRECTORY + '/image/' + str(i) + '.png', data_image)

    data_image = format_image(data_image)
    return data_image


def get_dataset(csv_path):
    data = pd.read_csv(csv_path)
    labels = []
    images = []
    count = 0
    total = data.shape[0]
    print('Total data : ' + str(total))
    for index, row in data.iterrows():
        emotion = emotion_to_vec(row['emotion'])
        image = data_to_image(row['pixels'], index)

        if image is not None:
            labels.append(emotion)
            images.append(image)

            count += 1
        print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    print("Total images: " + str(len(images)))

    np.save(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME), images)
    np.save(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME), labels)


if __name__ == '__main__':
    get_dataset(join(SAVE_DIRECTORY, DATASET_CSV_FILENAME))
    pass
