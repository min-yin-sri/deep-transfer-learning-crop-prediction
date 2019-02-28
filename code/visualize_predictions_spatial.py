import argparse
import getpass

import matplotlib
import sys
import pdb

import time

matplotlib.use('Agg')

from collections import defaultdict
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os
import numpy as np

from mpl_toolkits.basemap import Basemap
from unidecode import unidecode

import pandas as pd
import seaborn as sns

from os.path import isfile, join, getsize, expanduser, normpath, basename

from constants import GBUCKET, BASELINE_DIR, VISUALIZATIONS, LOCAL_DATA_DIR
from make_datasets import return_yield_file, return_filtered_regions_file

REGION_1 = "Region1"
REGION_2 = "Region2"
CROP = "Crop"
YEAR = "Year"

pdb.set_trace()
CLEAN_NAME = lambda r, l: unidecode(unicode(r.get(l,""), encoding='utf-8')).lower().translate(None, "'()/&-").strip()

USA_FIPS_CODES = {
    "29": "MO", "20": "KS", "31": "NE", "19": "IA", "38": "ND", "46": "SD",
    "27": "MN", "05": "AR", "17": "IL", "18": "IN", "39": "OH"
}


GET_FIPS = lambda r, l: USA_FIPS_CODES.get(r.get(l,""), "").lower()

t = time.localtime()
timeString  = time.strftime("%Y-%m-%d_%H-%M", t)

def return_shape_file(country):
    return join(LOCAL_DATA_DIR, "{}_shapefiles".format(country), "shape")

def get_test_data(NN_output_dir, input_data_dir):
    nn_data_dir = os.path.expanduser(os.path.join('/Users/burakuzkent/deep-transfer-learning-crop-prediction/es262-yield-africa/nnet_data/nnet_data', NN_output_dir))
    input_directory = os.path.expanduser(os.path.join('/Users/burakuzkent/deep-transfer-learning-crop-prediction/es262-yield-africa/datasets', input_data_dir))

    input_directory = '/Users/burakuzkent/deep-transfer-learning-crop-prediction/es262-yield-africa/datasets' + input_data_dir
    nn_data_dir = '/Users/burakuzkent/deep-transfer-learning-crop-prediction/es262-yield-africa/nnet_data' + NN_output_dir
    pdb.set_trace()
    test_labels_file = os.path.join(input_directory, 'test_yields.npz')
    test_labels = np.load(test_labels_file)['data']
    train_labels_file = os.path.join(input_directory, 'train_yields.npz')
    train_labels = np.load(train_labels_file)['data']

    test_keys_file = os.path.join(input_directory, 'test_keys.npz')
    test_keys = np.load(test_keys_file)['data']
    train_keys_file = os.path.join(input_directory, 'train_keys.npz')
    train_keys = np.load(train_keys_file)['data']

    train_prediction_path = os.path.join(nn_data_dir, 'train_predictions.npz')
    test_prediction_path = os.path.join(nn_data_dir, 'test_predictions.npz')
    test_predictions = np.load(test_prediction_path)['data']
    train_predictions = np.load(train_prediction_path)['data']
    test_labels = test_labels[:len(test_predictions)]
    train_labels = train_labels[:len(train_predictions)]
    test_keys = test_keys[:len(test_predictions)]
    train_keys = train_keys[:len(train_predictions)]

    print len(test_labels)
    print len(test_predictions)
    print len(test_keys)

    return test_labels, test_predictions, test_keys, train_labels, train_predictions, train_keys


def generate_spatial_error_plot(labels, predictions, keys, country, shapefile, input_data_dir, predtype, minrange, maxrange, compare=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if country == "argentina":
        map = Basemap(llcrnrlon=-74.0, llcrnrlat=-42., urcrnrlon=-53., urcrnrlat=-21., resolution='i')
        reg_name = lambda region: CLEAN_NAME(region, 'partido') + "_" + CLEAN_NAME(region, 'provincia')
        reg_filter = lambda region: True
    elif country == "brazil":
        map = Basemap(llcrnrlon=-74.0, llcrnrlat=-34., urcrnrlon=-34., urcrnrlat=6.,resolution='i')
        reg_name = lambda region: CLEAN_NAME(region, 'NM_MESO') + "_brasil"
        reg_filter = lambda region: True
    elif country == "usa":
        map = Basemap(llcrnrlon=-104.0, llcrnrlat=33., urcrnrlon=-80., urcrnrlat=49.,resolution='i')
        reg_name = lambda region: CLEAN_NAME(region, 'NAME') + "_" + GET_FIPS(region, 'STATEFP')
        reg_filter = lambda region: region.get('STATEFP',"") in USA_FIPS_CODES
    elif country == "southsudan":
        map = Basemap(llcrnrlon=23.593278, llcrnrlat=3.400894, urcrnrlon=35.746374, urcrnrlat=12.604212,resolution='i')
        reg_name = lambda region: CLEAN_NAME(region, 'ADMIN2') + "_" + CLEAN_NAME(region, 'ADMIN1')
        reg_filter = lambda region: True
    else:
        raise NotImplementedError()

    map.readshapefile(shapefile, "shapes")
    print keys

    patches = []
    data = []
    # pdb.set_trace()
    for info, shape in zip(map.shapes_info, map.shapes):
        info['ADMIN1'] = info['ADMIN1'].encode('ascii')
        info['ADMIN2'] = info['ADMIN2'].encode('ascii')
        key = reg_name(info)
        key_ixs = [l.startswith(key) for l in keys]
        yieldval = labels[key_ixs]
        predictval = predictions[key_ixs]
        if yieldval.shape[0] == 0 or predictval.shape[0] == 0 or yieldval.shape[0] != predictval.shape[0]:
            continue 
        if not reg_filter(info):
            continue

        print key
        patches.append(Polygon(np.array(shape), True))
        if compare is not None:
            error1 = np.abs(yieldval - predictval)
            error2 = np.abs(yieldval - compare[key_ixs])
            data.append((np.mean(error1 - error2)))
        else:
            data.append(np.mean(yieldval))

    cmapval = 'viridis' if compare is None else 'inferno'
    cmaprange = Normalize(minrange,maxrange)
    p = PatchCollection(patches, facecolor='m', edgecolor='k', linewidths=1., zorder=2, cmap=cmapval, norm=cmaprange)
    p.set_array(np.array(data))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax)

    title = os.path.basename(os.path.normpath(input_data_dir)) + '_' + timeString
    savedir = os.path.join(GBUCKET, VISUALIZATIONS, getpass.getuser())
    savename = join(savedir, 'prediction_{}_{}.png'.format(predtype, title))

    plt.savefig(savename)

def main():
    parser = argparse.ArgumentParser(description="Generate spatial visualizations of model error")
    parser.add_argument("NN_output_name", help="Choose the directory that contains all the NN outputs")
    parser.add_argument("data_input_name", help="Choose the directory that contains the input dataset")
    parser.add_argument("--compare_NN", default=None, help="Choose the directory containing a second NN output set for comparison")
    parser.add_argument("country", help="Country for analysis")
    parser.add_argument("range_min", type=float, default=-10., help="Min value for color scheme")
    parser.add_argument("range_max", type=float, default=10., help="Max value for color scheme")
    args = parser.parse_args()

    country = args.country
    boundaries_shp = return_shape_file(country)
    NN_output_name = args.NN_output_name
    data_input_name = args.data_input_name
    range_min = args.range_min
    range_max = args.range_max

    test_labels, test_predictions, test_keys, train_labels, train_predictions, train_keys = \
        get_test_data(NN_output_name, data_input_name)

    if args.compare_NN:
        test_labels_2, test_predictions_2, test_keys_2, train_labels_2, train_predictions_2, train_keys_2 = \
            get_test_data(args.compare_NN, data_input_name)
        generate_spatial_error_plot(test_labels, test_predictions, test_keys,
                                    country, boundaries_shp, data_input_name, "test", 
                                    range_min, range_max, compare=test_predictions_2)
    else:
        generate_spatial_error_plot(test_labels, test_predictions, test_keys,
                                    country, boundaries_shp, data_input_name, "test",
                                    range_min, range_max)
    #generate_spatial_error_plot(train_labels, train_predictions, train_keys,
    #                            country, boundaries_shp, data_input_name, "train")

if __name__ == '__main__':
    main()
