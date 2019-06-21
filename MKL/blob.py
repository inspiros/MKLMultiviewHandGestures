import collections
import array
import numpy as np

def read_binary_blob(filename):
    #
    # Read binary blob file from C3D
    # INPUT
    # filename    : input filename.
    #
    # OUTPUT
    # s           : a 1x5 matrix indicates the size of the blob
    #               which is [num channel length height width].
    # blob        : a 5-D tensor size num x channel x length x height x width
    #               containing the blob data.
    # read_status : a scalar value = 1 if sucessfully read, 0 otherwise.


    # precision is set to 'single', used by C3D

    # open file and read size and data buffer
    # [s, c] = fread(f, [1 5], 'int32');
    read_status = 1
    blob = collections.namedtuple('Blob', ['size', 'data'])

    f = open(filename, 'rb')
    s = array.array("i") # int32
    s.fromfile(f, 5)

    if len(s) == 5 :
        m = s[0]*s[1]*s[2]*s[3]*s[4]

        # [data, c] = fread(f, [1 m], precision)
        data_aux = array.array("f")
        data_aux.fromfile(f, m)
        data = np.array(data_aux.tolist())

        if len(data) != m:
            read_status = 0;

    else:
        read_status = 0;

    # If failed to read, set empty output and return
    if not read_status:
        s = []
        blob_data = []
        b = blob(s, blob_data)
        return s, b, read_status

    # reshape the data buffer to blob
    # note that MATLAB use column order, while C3D uses row-order
    # blob = zeros(s(1), s(2), s(3), s(4), s(5), Float);
    blob_data = np.zeros((s[0], s[1], s[2], s[3], s[4]), np.float32)
    off = 0
    image_size = s[3]*s[4]
    for n in range(0, s[0]):
        for c in range(0, s[1]):
            for l in range(0, s[2]):
                # print n, c, l, off, off+image_size
                tmp = data[np.array(range(off, off+image_size))];
                blob_data[n][c][l][:][:] = tmp.reshape(s[3], -1);
                off = off+image_size;


    b = blob(s, blob_data)
    f.close()
    return s, b, read_status


def load_np_array(filename):
	# Load numpy array from c3d data file
	s, b, r = read_binary_blob(filename)
	res = np.array([b[1][0][0][0][0][0]])
	for i in range (1, len(b[1][0])):
		res = np.concatenate((res, np.array([b[1][0][i][0][0][0]])), axis = 0)
	return res


def get_np_array(bblob):
	# Get numpy array from c3d blob
	res = np.array([bblob[1][0][0][0][0][0]])
	for i in range (1, len(bblob[1][0])):
		res = np.concatenate((res, np.array([bblob[1][0][i][0][0][0]])), axis = 0)
	return res