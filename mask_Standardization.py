# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
from collections import Counter

def get_filenames(dataset_dir):
    photo_filenames = []
    g = os.walk(dataset_dir)
    for path,dir_list,file_list in g:
        for file_name in file_list:
            if file_name.endswith('.png'):
                photo_filenames.append(os.path.join(path, file_name))
    return photo_filenames

if __name__ == '__main__':

    paths = get_filenames('./rendering_1/masks')

    for path in paths:
        print(path)
        img = Image.open(path)
        # print(len(img.split()))
        array = np.array(img)
        result = array.copy()

        cnt = Counter(array.flatten())
        ids = []
        for id in cnt:
            if id != 0: 
                ids.append(id) 
        print(ids)

        num = 0
        for a in ids:
            result[:] = 0
            mask = array == a


            result[mask] = 255

            img = Image.fromarray(result, mode= 'RGB')
            # img.show()
            img.save(path.replace('.png', '_Bearing_'+ str(num)+ '.png').replace('masks','masks_standard'))
            num += 1



    # img = Image.open('./Mask4Seg_1_2020-12-17-16-40-30.png')
    # # print(len(img.split()))
    # array = np.array(img)

    # cnt = Counter(array.flatten())
    # print(cnt)
    # ids = []
    # for id in cnt:
    #     if id != 0: 
    #         ids.append(id) 
    # print(ids)


    # result = array.copy()
    # num = 0
    # for a in ids:
    #     result[:] = 0
    #     mask = array == a


    #     result[mask] = 255

    #     img = Image.fromarray(result, mode= 'RGB')
    #     img.show()