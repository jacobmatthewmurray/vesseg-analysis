#! /usr/bin/python

import xml.etree.ElementTree as ET
import numpy as np
import os
import pathlib
import sys

from multiprocessing import Pool
from PIL import Image

from skimage.draw import polygon


def xml_tree_roi_from_vsi(vsi_file):

    xml_identifier_start = '<?xml '
    xml_identifier_stop = '</siXML>'

    with open(vsi_file, errors='ignore') as f:
        vsi_string = f.read()

    vsi_string = vsi_string.replace(u'\u0000', '')

    xml_start = vsi_string.find(xml_identifier_start)
    xml_stop = vsi_string.find(xml_identifier_stop)

    if xml_start == -1 or xml_stop == -1:
        print('{} does not contain ROI XML Element.'.format(vsi_file))
        return None

    xml_string = vsi_string[xml_start:xml_stop+len(xml_identifier_stop)]

    return ET.fromstring(xml_string)


def get_polygon_vectors(xml_tree):

    polygons = list()

    for polygon in xml_tree.findall(".//Property[@Key='measure:ShapePoints']"):

        coordinates = [float(c.text) for c in polygon.findall("./Component/CdVec2/double")]
        polygons.append(np.reshape(coordinates, (2, -1), order='F'))

    return polygons


def img_dim_from_metadata_xml(metadata_xml):

    requested_attributes = [
        'PhysicalSizeX',
        'PhysicalSizeY',
        'PhysicalSizeXUnit',
        'PhysicalSizeYUnit',
        'SizeX',
        'SizeY'
    ]

    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

    tree = ET.parse(metadata_xml)
    elem = tree.find(".//ome:Image[@ID='Image:0']/ome:Pixels", ns)

    return_dict = dict()

    for a in requested_attributes:
        return_dict[a] = elem.get(a)

    return return_dict


def make_binary_masks(vsi_file):

    try:

        vsi_file_label = os.path.basename(vsi_file).replace('.vsi', '')
        vsi_file_label = vsi_file_label.replace('.', '_').replace(' ', '_')

        xml_roi = xml_tree_roi_from_vsi(vsi_file)

        
        if not xml_roi:
            return None


        meta_path = os.path.join(os.path.dirname(vsi_file), '_' + os.path.splitext(os.path.basename(vsi_file))[0] + '_')
        mask_path = os.path.join(meta_path, 'masks')
        pathlib.Path(mask_path).mkdir(exist_ok=True, parents=True)

        img = Image.open(os.path.join(meta_path, 'images', vsi_file_label + '.tif'))
        img_array = np.array(img)
        img_array_x = img_array.shape[0] - 1

        roll_cnt = 0

        while np.array_equal(img_array[img_array_x, roll_cnt], np.array([255, 255, 255])):
            roll_cnt += 1

        # img_dim = img_dim_from_metadata_xml(os.path.join(meta_path, vsi_file_label + '__metadata.xml'))
        polygons = get_polygon_vectors(xml_roi)

        #TODO add metadata

        consolidated_mask = np.zeros((img.size[1], img.size[0]), dtype='uint8')
        roi_size = list()

        for i, p in enumerate(polygons):
            mask = np.zeros((img.size[1], img.size[0]), dtype='uint8')
            rr, cc = polygon(np.ceil(p[1]), np.ceil(p[0]))
            mask[rr, cc] = 1
            mask = np.roll(mask, roll_cnt)
            roi_size.append(np.sum(mask))
            consolidated_mask += mask

            image = Image.fromarray(mask)
            image.save(os.path.join(mask_path, vsi_file_label + '__mask_' + str(i) + '.png'), 'PNG', optimize=True)

            # tif_file = os.path.join(os.path.dirname(meta_path), 'mask_' + str(i) + '.tif')
            # tf.imwrite(tif_file, mask, photometric='mask', compress=6)

        consolidated_mask[consolidated_mask > 2] = 2
        image = Image.fromarray(consolidated_mask)
        image.save(os.path.join(mask_path, vsi_file_label + '__mask_consolidated.png'), 'PNG', optimize=True)

        idx_lumen = np.argmax(roi_size)
        old_mask_name = vsi_file_label + '__mask_' + str(idx_lumen) + '.png'
        new_mask_name = vsi_file_label + '__mask_' + str(idx_lumen) + '_full_lumen.png'
        os.rename(os.path.join(mask_path, old_mask_name), os.path.join(mask_path, new_mask_name))

    except Exception as e:
        print(e)

def make_binary_masks_directory(directory):

    for root, dir, file in os.walk(directory):
        for f in file:
            if f.endswith('.vsi'):
                try:
                    make_binary_masks(os.path.join(root, f))
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    
    CORES = 10
    
    path_list = list()
    
    for root, dir, file in os.walk('/path/'):
        for f in file:
            if f.endswith('.vsi') and 'temp' not in root:
                path_list.append(os.path.join(root, f))
    
    CHUNKS = len(path_list) // CORES
    
    with Pool(processes=15) as pool:
        pool.map(make_binary_masks, path_list)

