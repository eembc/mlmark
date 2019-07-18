import json
import csv
import sys
import os
from eelib import paths

# if len(sys.argv) < 2:
#   print("csv file containing detection results is required.")
#    sys.exit(1)

#csv_file = sys.argv[1]

def parse_ssd_detection(detection_file):

    validation_data = {}
    validation_data_output = []

    with open(detection_file, 'r') as csvfile:
        csv_data=csv.reader(csvfile, delimiter=',')

        for row in csv_data:
           image_name = row[0]
           if not image_name in validation_data.keys():
                validation_data[image_name] = []
       
           detection = {'class': None, 'box': [], 'score': None}
           detection['class'] = float(row[1])
           detection['box'] = [float(row[2]),float(row[3]),float(row[4]),float(row[5])]
           detection['score'] = float(row[-1])
           validation_data[image_name].append(detection)

    for key in validation_data.keys():
        detections = [key,validation_data[key]]
        validation_data_output.append(detections)
    return validation_data_output

#    output_path=os.path.join(paths.TARGETS,"openvino_ubuntu","workloads","ssdmobilenet","ssd_detection_results.json")        
#    with open(output_path,'w') as fid:
#        json.dump(validation_data_output, fid, indent=4)
#        print("output printed to {}".format(output_path))
