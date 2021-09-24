
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from multiprocessing import Pool
# from scipy import ndimage
import numpy as np
import os
import pathlib


def crop_image_to_com_artery(image_path, mask_path, bounding_box):

    mask = np.array(Image.open(mask_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # adjust bounding box

    bb = np.min(list(zip(bounding_box, mask.shape)), axis=1)

    if tuple(bounding_box) != tuple(bb):
        print('Bounding box reduced to {}.'.format(bb))

    # boundary box around mask approach

    non_zero_rows, non_zero_cols = np.nonzero(mask)

    row_min = np.min(non_zero_rows)
    row_max = np.max(non_zero_rows)
    col_min = np.min(non_zero_cols)
    col_max = np.max(non_zero_cols)

    center = np.array([(row_max + row_min) / 2, (col_max + col_min) / 2])

    top_left = center - bb/2
    top_left = top_left.astype(int)

    # legacy com approach

    # mask_ones = np.copy(mask)
    # mask_ones[mask_ones > 0] = 1

    # com = ndimage.measurements.center_of_mass(mask_ones)
    # com = np.floor(com)

    # top_left = com - bb/2
    # top_left = top_left.astype(int)

    for i in range(len(bb)):
        if top_left[i] < 0:
            top_left[i] = 0
        if top_left[i] + bb[i] > mask.shape[i]:
            top_left[i] = mask.shape[i] - bb[i]

    mask = mask[top_left[0]:top_left[0]+bb[0], top_left[1]:top_left[1]+bb[1]]

    bb_string = '_'.join(bb.astype(str))

    image = Image.fromarray(mask)
    image.save(os.path.join(os.path.dirname(mask_path), image_name + '__mask_cropped_' + bb_string + '.png'),
               'PNG', optimize=True)

    image = Image.open(image_path)
    image = image.crop((top_left[1], top_left[0], top_left[1]+bb[1], top_left[0]+bb[0]))
    image.save(os.path.join(os.path.dirname(image_path), image_name + '__cropped_' + bb_string + '.png'),
               'PNG', optimize=True)


def rename_metadata_xml(root_directory):

    for root, dir, file in os.walk(root_directory):
        for f in file:
            if f.endswith('.vsi') and 'temp' not in root:
                try:
                    vsi_file_label = f.replace('.vsi', '')
                    vsi_file_label = vsi_file_label.replace('.', '_').replace(' ', '_')

                    old_file_name = 'metadata.xml'
                    new_file_name = vsi_file_label + '__metadata.xml'

                    stack_folder = '_' + f.replace('.vsi', '') + '_'

                    os.rename(os.path.join(root, stack_folder, old_file_name),
                              os.path.join(root, stack_folder, new_file_name))
                except Exception as e:
                    print(e)


def delete_rename_converted_tiff(root_directory):

    for root, dir, file in os.walk(root_directory):
        for f in file:
            if f.endswith('.vsi') and 'temp' not in root:
                vsi_file_label = f.replace('.vsi', '')
                vsi_file_label = vsi_file_label.replace('.', '_').replace(' ', '_')
                stack_folder = '_' + f.replace('.vsi', '') + '_'

                tiffs = [i for i in os.listdir(os.path.join(root, stack_folder)) if i.endswith('.tif')]
                tiffs_size = [os.stat(os.path.join(root, stack_folder, i)).st_size for i in tiffs]

                max_tiff_idx = np.argmax(tiffs_size)

                pathlib.Path(os.path.join(root, stack_folder, 'images')).mkdir(exist_ok=True)
                os.rename(os.path.join(root, stack_folder, tiffs[max_tiff_idx]),
                          os.path.join(root, stack_folder, 'images', vsi_file_label + '.tif'))

                del tiffs[max_tiff_idx]

                for t in tiffs:
                    os.remove(os.path.join(root, stack_folder, t))


def get_vsi_info_from_file(vsi_file_path):

    assert vsi_file_path.endswith('.vsi'), '{} is not a .vsi file.'.format(vsi_file_path)

    vsi_name = os.path.splitext(os.path.basename(vsi_file_path))[0]

    vsi_stack_folder = '_' + vsi_name + '_'
    vsi_label =  vsi_name.replace('.', '_').replace(' ', '_')

    return vsi_label, vsi_stack_folder


def vsi_directory_parser(root_directory, contain_tag):

    for root, dir, file in os.walk(root_directory):
        for f in file:
            if f.endswith('.vsi'):

                vsi_label, vsi_stack_folder = get_vsi_info_from_file(f)

                req_paths = {'images': [], 'masks': []}

                for req in req_paths:
                    if os.path.isdir(os.path.join(root, vsi_stack_folder, req)):
                        req_paths[req] = [os.path.join(root, vsi_stack_folder, req, i)
                                          for i in os.listdir(os.path.join(root, vsi_stack_folder, req))
                                          if contain_tag[req] in i]

                yield req_paths


def crop_image_to_com_artery_directory(root_directory, bounding_box):

    for paths in vsi_directory_parser(root_directory, {'images': '.tif', 'masks': '__mask_consolidated'}):

        if paths['images'] and paths['masks']:
            print('Cropping to {} for image {} and mask {}.'.format(bounding_box, paths['images'], paths['masks']))
            crop_image_to_com_artery(paths['images'][0], paths['masks'][0], bounding_box)


def crop_resize(my_paths):
    try:
        image_path, mask_path, destination_path = my_paths

        mask = np.array(Image.open(mask_path))
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        non_zero_rows, non_zero_cols = np.nonzero(mask)

        row_min = np.min(non_zero_rows)
        row_max = np.max(non_zero_rows)
        col_min = np.min(non_zero_cols)
        col_max = np.max(non_zero_cols)

        padding = int(0.25 * ((row_max - row_min + col_min - col_min) / 2))

        col_length = 2*padding + col_max - col_min
        row_length = 2*padding + row_max - row_min

        bb = np.array([row_length, col_length])
        bb = np.min(list(zip(bb, mask.shape)), axis=1)

        top_left = np.array([row_min - padding, col_min - padding])
        top_left = top_left.astype(int)

        for i in range(len(bb)):
            if top_left[i] < 0:
                top_left[i] = 0
            if top_left[i] + bb[i] > mask.shape[i]:
                top_left[i] = mask.shape[i] - bb[i]

        png_info_dict = {
            'original_y': mask.shape[0],
            'original_x': mask.shape[1],
            'crop_top_left_x': top_left[0],
            'crop_top_left_y': top_left[1],
            'crop_dim_x': bb[0],
            'crop_dim_y': bb[1]
        }

        png_info = PngInfo()

        for key, value in png_info_dict.items():
            png_info.add_text(key, str(value))

        mask = mask[top_left[0]:top_left[0] + bb[0], top_left[1]:top_left[1] + bb[1]]

        image = Image.fromarray(mask)
        image = image.resize((2048, 2048), resample=Image.BICUBIC)
        pathlib.Path(os.path.join(destination_path, 'masks')).mkdir(exist_ok=True, parents=True)
        image.save(os.path.join(destination_path, 'masks', image_name + '__mask_c_r.png'),
                   'PNG', optimize=True, pnginfo=png_info)

        image = Image.open(image_path)
        image = image.crop((top_left[1], top_left[0], top_left[1] + bb[1], top_left[0] + bb[0]))
        image = image.resize((2048, 2048), resample=Image.BICUBIC)
        pathlib.Path(os.path.join(destination_path, 'images')).mkdir(exist_ok=True, parents=True)
        image.save(os.path.join(destination_path, 'images', image_name + '__c_r.png'),
                   'PNG', optimize=True, pnginfo=png_info)

    except Exception as e:
        print(e)


if __name__ == "__main__":

    conversion_list = []
    
    for root, dir, files in os.walk('/media/jacob/data/jacob/vesseg/test'):
    
        for f in files:
            if f.endswith('.tif'):
                    img_path = os.path.join(root, f)
                    msk_path = img_path.replace('/images', '/masks').replace('.tif', '__mask_consolidated.png')
                    des_path = '/media/jacob/data/jacob/vesseg/test'
    
                    if msk_path not in mask_list:
                        conversion_list.append([img_path, msk_path, des_path])
    
    with Pool(processes=15) as pool:
        pool.map(crop_resize, conversion_list)
    










