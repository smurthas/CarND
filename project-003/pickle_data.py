import sys

from PIL import Image
import cv2
import numpy as np
import pickle
import time

def process_image(im):
    im = np.array(im.resize((200, 160)))
    im = np.reshape(im, (160, 200, 3))
    im = im[64:130, 0:200]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    return im

def image_file_to_array(fn):
    return process_image(Image.open(fn))

def should_drop(angle, array, min_match, match_to=0.0001):
    if min_match < 1:
        return False
    if len(array) < min_match:
        return False
    m = 1/match_to
    for ang in array:
        if int(angle*m) != int(ang*m):
            return False
    return True

def read_in_data(path_to_datas, p_filename, limit=1000000, min_angle=0, max_angle=0.95, drop_multis=0, lr=0.25):
    per_file = 50000

    csv_filename = path_to_datas + '/driving_log.csv'
    img_list = [line.split(',') for line in open(csv_filename)][1:]

    data = {}
    data['features'] = []
    data['angles'] = []

    i = 0
    file_number = 0
    total = min(len(img_list), limit)
    start_time = time.time()

    prev_angles = []

    for line in img_list:
        i += 1
        if i > limit:
            break
        if i % per_file == 0:
            elapsed_time = time.time() - start_time
            ETA = elapsed_time / i * (total - i)
            print('%d/%d'%(i, total), ' %ds remaining'%int(ETA))
            fn = p_filename + '_' + str(file_number) + '.p'
            pickle.dump(data, open(fn, "wb"))
            data = {}
            data['features'] = []
            data['angles'] = []
            file_number += 1


        fn_center = path_to_datas + '/IMG/' +line[0].split('/')[-1:][0]
        fn_left = path_to_datas + '/IMG/' +line[1].split('/')[-1:][0]
        fn_right = path_to_datas + '/IMG/' +line[2].split('/')[-1:][0]
        #print(fn)
        angle = float(line[3])
        if abs(angle) < min_angle or abs(angle) > max_angle:
            print('discarding angle > 0.5: ', angle)
            continue

        if drop_multis > 0:
            if should_drop(angle, prev_angles, drop_multis):
                print('discarding repeat angle:', angle, prev_angles)
                continue
            if len(prev_angles) == drop_multis:
                prev_angles.pop(0)
            prev_angles.append(angle)

        img_center = image_file_to_array(fn_center)
        if lr > 0:
            img_left = image_file_to_array(fn_left)
            img_right = image_file_to_array(fn_right)
        if len(img_center) < 66 or len(img_center[0]) < 200 or \
            (lr>0 and (len(img_left) < 66 or len(img_left[0]) < 200)) or \
            (lr>0 and (len(img_right) < 66 or len(img_right[0]) < 200)):
            print('bad image?', img_center, img_left, _img_right)
            continue

        data['features'].append(img_center)
        data['angles'].append(angle)
        if lr>0:
            data['features'].append(img_left)
            data['angles'].append(angle+lr)
            data['features'].append(img_right)
            data['angles'].append(angle-lr)

    fn = p_filename + '_' + str(file_number) + '.p'
    pickle.dump(data, open(fn, "wb"))

    return data

if __name__ == '__main__':
    read_in_data(sys.argv[1], sys.argv[2], min_angle=0.05, max_angle=0.99, drop_multis=3, lr=0)
    #read_in_data(sys.argv[1], sys.argv[2])

