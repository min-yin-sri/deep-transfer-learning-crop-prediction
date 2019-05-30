
import sys
import numpy as np
import pickle
import random
import math
import argparse
import logging
import os
import glob
import csv

PATH = "/root/bucket3/textual_global_feature_vectors"
COORDINATES_CSV_FILENAME = "All_Image_Coordinates_2.csv"
SOUTH_SUDAN_CSV_FILENAME = "South_Sudan_Coordinates_.csv"

# The Minimum of South Sudan Latitude
LAT_MIN = 3
# The Maximum of South Sudan Latitude
LAT_MAX = 13
# The Minimum of South Sudan Longitude
LON_MIN = 24
# The Maximum of South Sudan Longitude
LON_MAX = 36


desc = """ Create a csv file that filter out all the none South Sudan coordinates entries from the all coordinates file.
TODO:  
"""


if __name__ == "__main__":
  parser = argparse.ArgumentParser( description = desc )
  parser.add_argument( "--csv_dir", type = str, default = PATH, help = "Directory that holds the original all coordinates csv files" )
  parser.add_argument( "--all_csv_file", type = str, default = COORDINATES_CSV_FILENAME, help = "All cordinate csv file name" )
  parser.add_argument( "--output_dir", "-o", type = str, default = PATH, help = "Output directory of the South Sudan coordinates csv files" )
  parser.add_argument( "--south_sudan_csv_file", type = str, default = SOUTH_SUDAN_CSV_FILENAME, help = "Image wild card in each sequence" )
  parser.add_argument( "--verbosity", "-v", action="count", default = 1, help = "Verbosity level" )
  args = parser.parse_args()
    
  # set logging level 
  console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
  logging.basicConfig( level = console_level, format = '[%(levelname)s] %(message)s' )

  input_file = os.path.join( args.csv_dir, args.all_csv_file )
  logging.info("Input file is at %s" % input_file)

  with open(input_file, 'rb') as f:
    reader = csv.reader(f)
    all_csv_list = list(reader)

  logging.info("All csv file has %d entries" % len(all_csv_list))
  logging.info("The first line of all csv file: %s %s %s" % (all_csv_list[0][0], all_csv_list[0][3], all_csv_list[0][4]) )
  logging.info("The first line of all csv file: %s %s %s" % (all_csv_list[1][0], all_csv_list[1][3], all_csv_list[1][4]) )

  if not os.path.exists( args.output_dir ):
    logging.info( "Creating folder: %s" % args.output_dir )
    os.makedirs( args.output_dir )

  output_file = os.path.join( args.output_dir, args.south_sudan_csv_file )
  logging.info("Output file is at %s" % output_file)

  south_sudan_csv_list[0] = all_csv_list[0]
  index = 1
  for entry in all_csv_list:
    if (float(entry[3]) >= LAT_MIN and float(entry[3]) <= LAT_MAX 
        and float(entry[4]) >= LON_MIN and float(entry[4]) <= LON_MAX):
        south_sudan_csv_list[index] = entry
        index = index + 1

  logging.info( "South Sudan csv file has %d entries" % len(south_sudan_csv_list))
  logging.info( "---" )
