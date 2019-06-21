import os
import numpy as np
import blob
from MKL.algorithms import EasyMKL


def load_c3d_features_from_folder(folder, layer='fc6'):
	data_dict = {}
	for fc_file in os.listdir(folder):
		if fc_file[len(fc_file) - 4:] == '.' + layer:
			fc_file_path = os.path.join(folder, fc_file)
			data_dict.update({fc_file[:len(fc_file) - 4] : blob.load_np_array(fc_file_path)})
	return data_dict

