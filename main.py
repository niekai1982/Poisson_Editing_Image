import cv2
import numpy as np
import os
import poissonimageediting as poisson
from time import time
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))

mask_path = os.path.join(path, 'rendering_part_1/masks')
sample_path = os.path.join(path, 'rendering_part_1/samples')
target_path = os.path.join(path, 'rendering_part_0/target')
mask4merge_path = os.path.join(path, 'rendering_part_1/masks4merge')

mask_files = os.listdir(mask_path)
sample_files = os.listdir(sample_path)
mask4merge_files = os.listdir(mask4merge_path)
# target_files = [ file for file in os.listdir(target_path) if file.endswith('.JPG')]
target_files = [ 'target.jpg']

sample_files_dict = {}
mask_files_dict = {}
mask4merge_dict = {}

for file_name in mask_files:
    time_id = os.path.splitext(file_name)[0].split('_')[-1]
    mask_files_dict[time_id] = file_name

for file_name in sample_files:
    time_id = os.path.splitext(file_name)[0].split('_')[-1]
    sample_files_dict[time_id] = file_name

for file_name in mask4merge_files:
    time_id = os.path.splitext(file_name)[0].split('_')[-1]
    mask4merge_dict[time_id] = file_name


mix_dir = "{0}/rendering_part_1/mix".format(path)
overlap_dir = "{0}/rendering_part_1/overlap".format(path)
data_dir = "{0}/rendering_part_1/data".format(path)

if(not(os.path.exists(mix_dir))):
    os.mkdir(mix_dir)
if(not(os.path.exists(overlap_dir))):
    os.mkdir(overlap_dir)
if(not(os.path.exists(data_dir))):
    os.mkdir(data_dir)

scale = 1.2

# for idx in range(len(sample_files)):
for tar_file in target_files:
    for idx in range(len(sample_files)):
        poisson.progress_bar(idx, len(sample_files))
        shift_x = np.random.random()
        shift_y = np.random.random()

        src_file_name = sample_files[idx]
        # target_file_name = tar_file
        time_id = os.path.splitext(src_file_name)[0].split('_')[-1]
        mask_file_name = mask_files_dict[time_id]
        mask4merge_file_name = mask4merge_dict[time_id]
        # target_file_name = "target.jpg"

        ### 1.  load images
        src = np.array(cv2.imread(os.path.join(sample_path, src_file_name), 1) / 255.0, dtype=np.float32)
        target  = np.array(cv2.imread(os.path.join(target_path, tar_file), 1) / 255.0, dtype=np.float32)
        # target = cv2.resize(target, (int(target.shape[1] / scale), int(target.shape[0] / scale)))
        target = cv2.resize(target, (520, 520))

        shift_x = int(shift_x * (target.shape[1] - src.shape[1]))
        shift_y = int(shift_y * (target.shape[0] - src.shape[0]))

        target = target[shift_y:shift_y + src.shape[0], shift_x:shift_x + src.shape[1], :]
        # target = target[:src.shape[0], :src.shape[1],:]
        mask  = np.array(cv2.imread(os.path.join(mask_path, mask_file_name), 0), dtype=np.uint8)
        mask4merge  = np.array(cv2.imread(os.path.join(mask4merge_path, mask4merge_file_name), 0), dtype=np.uint8)

        blended, overlaped = poisson.poisson_blend(src, mask4merge/255.0, target, "mix", data_dir)

        overlaped_file_name = os.path.splitext(tar_file)[0] + '_' + "overlap" + '_'  + str(idx) + '_' + time_id + '.jpg'
        mix_file_name = os.path.splitext(tar_file)[0] + '_' + "mix" + '_' + str(idx) + '_' + time_id + '.jpg'

        cv2.imwrite(os.path.join(mix_dir, mix_file_name), blended)
        cv2.imwrite(os.path.join(overlap_dir, overlaped_file_name), overlaped)

    # plt.subplot(1, 4, 1)
    # plt.imshow(src)
    # plt.subplot(1, 4, 2)
    # plt.imshow(mask)
    # plt.subplot(1, 4, 3)
    # plt.imshow(mask4merge)
    # plt.subplot(1, 4, 4)
    # plt.imshow(target)
    # plt.show()
    #
    #





