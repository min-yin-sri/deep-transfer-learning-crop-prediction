
import sys
import numpy as np
import gensim
import pandas as pd
import pickle
import random
import math
from math import sin, cos, sqrt, atan2, radians
import argparse
import logging
import os
import glob
import csv


PATH = "/root/bucket3/textual_global_feature_vectors"
POVERTY_GROUND_TRUTH_FILENAME = "wealth_index_cluster_locations_2017_08.csv"
COORDINATES_CSV_FILENAME = "Africa_Image_Coordinates.csv" #"All_Image_Coordinates_2.csv"
SOUTH_SUDAN_CSV_FILENAME = "South_Sudan_Coordinates.csv" #"Ethiopia_Coordinates.csv" 
SOUTH_SUDAN_GROUND_TRUTH_FILENAME = "South_Sudan_Grouth_Truth.csv"
ETHIOPIA_COODINATES_FILENAME = "Ethiopia_Coordinates.csv"
ETHIOPIA_GROUTH_TRUTH_FILENAME = "Ethiopia_Grouth_Truth.csv"

# The distance in km to check within
MARGIN = 10

# The number of km in one degree of latitude
LAT_KM = 110.574

# The number of km in one degree of longitude
LON_KM = 111.320

# Returns the embeddings and the average embedding of an array of articles
def get_embeddings_average(array, model):
    if len(array) == 0:
        return [], None
    embeddings = []
    for i in array:
        embeddings.append(model.docvecs[get_title(i[1])])

    av = []
    for i in range(len(embeddings[0])):
        sum_ = 0
        for j in embeddings:
            sum_ += j[i]
        sum_ /= float(len(embeddings))
        av.append(sum_)

    return embeddings, av

# Given coordinates a, b in deg, return the distance between a and b in km         
def compute_distance(c1, c2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(c1[0])
    lon1 = radians(c1[1])
    lat2 = radians(c2[0])
    lon2 = radians(c2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

desc = """ Build training data. For each entry in grouth truth, use coordiantes csv find the N nearest articles, stack the article's numpy features as the feature of this entry.
TODO:  
"""


if __name__ == "__main__":
  parser = argparse.ArgumentParser( description = desc )
  parser.add_argument( "number", type = "count", default = 10, help = "Number of closest articles to be found." )
  parser.add_argument( "--data_dir", type = str, default = PATH, help = "Directory that holds all the necessary data files" )
  parser.add_argument( "--coordinate_file", type = str, default = ETHIOPIA_COODINATES_FILENAME, help = "Coordinates csv file name" )
  parser.add_argument( "--ground_truth_file", type = str, default = ETHIOPIA_GROUTH_TRUTH_FILENAME, help = "Ground truth file name" )
  parser.add_argument( "--output_dir", "-o", type = str, default = PATH, help = "Output directory of created training feature files" )
  parser.add_argument( "--verbosity", "-v", action="count", default = 1, help = "Verbosity level" )
  args = parser.parse_args()
    
  # set logging level 
  console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
  logging.basicConfig( level = console_level, format = '[%(levelname)s] %(message)s' )

  coordinate_input_file = os.path.join( args.data_dir, args.coordinate_file )
  logging.info("Coordiantes file is at %s" % coordinate_input_file)

  with open(coordinate_input_file, 'rb') as cf:
    creader = csv.reader(cf)
    coordinate_list = list(creader)

  logging.info("coordinate csv file has %d entries" % len(coordinate_list))
  logging.info("The first line of coordinate csv file: %s %s %s" % (coordinate_list[0][0], coordinate_list[0][3], coordinate_list[0][4]) )
  logging.info("The first line of coordinate csv file: %s %s %s" % (coordinate_list[1][0], coordinate_list[1][3], coordinate_list[1][4]) )

  ground_trutch_input_file = os.path.join( args.data_dir, args.ground_truth_file )
  logging.info("Ground truth file is at %s" % ground_truth_input_file)

  with open(ground+truth_input_file, 'rb') as gf:
    greader = csv.reader(gf)
    ground_truth_list = list(greader)

  logging.info("ground truth csv file has %d entries" % len(ground_truth_list))
  logging.info("The first line of ground truth csv file: %s %s %s" % (ground_truth_list[0][0], ground_truth_list[0][8], ground_truth_list[0][9]) )
  logging.info("The first line of ground truth csv file: %s %s %s" % (ground_truth_list[1][0], ground_truth_list[1][8], ground_truth_list[1][9]) )

  logging.info( "---" )
