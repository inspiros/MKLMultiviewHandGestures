# MKL evaluation on MultiviewHandGestures
Dedicated to run on FC features formatted by code in `/media/data3/tranhoangnhat/c3d_luanvan`
Download extracted features folders to local computer or push this code folder to server.
### Input
FC features folder
```python
dataset
│	RGB_K1_K1
│	...
└───RGB_K3_K3 '''[Modality]_[Kinect train]_[Kinect test]'''
│	│	Binh
│	└───Giang '''[Test subject]'''
│		└───feature
│			│	iter_100
│			│	...
│			└───iter_800 '''iter_[num of iteration]'''
│				│	Binh
│				└───Giang
│				│	└───1 '''[gesture number]'''
│				│		└───1 '''[setup index]'''
│				│			└───000001.fc6 '''[FC6 feature]'''
│				│			└───000001.fc7 '''[FC7 feature]'''
│				│			└───000001.prob '''[Probability]'''
│				│	...
│	Depth_K1_K1
│	Depth_K2_K2   
└───Depth_K3_K3
│	└─── '''<similar format>'''
│	...
```
### Output

Confusion matrices, classification reports and f1-score csv of
- SVM on single modality
- SVM on concatenated normalized features
- MKL

In the following format:
```python
result
└───confusion #contains confusion matrices
│	└───RGB_800 Depth_1600 '''RGB_[num iter rgb] Depth_[num iter depth]'''
│		└───K3_K3_laplacian '''[Kinect train]_[Kinect test]_[kernels]'''
│			└───Binh
│			│	...
│			└───Thuan '''[Test subject of leave one out]'''
│			│	└─── '''<Confusion matrix images and csv files of result testing on each subject>'''
│			└─── '''<Confusion matrix images and csv files of overall result>'''
└───report #contain classification reports
	└───RGB_800 Depth_1600
		└───K3_K3_laplacian
			└───Binh
			│	...
			└───Thuan
			│	└─── '''<classification report csv of result testing on each subject>'''
			└─── '''<classification report csv of overall result>'''
```
## How to use
### Dependencies
Embedded MKL codes, no need to download library. But possibly requires `sklearn`, `pandas`, `matplotlib`
### Configure parameters
Firstly, set the dataset folder in file `configs.py`:
```python
DATASET_ROOT = '/home/path/to/folder/containing/features'
```
[Optional] Modify kernels, SVM, MKL parameters in file `configs.py`. There are predefined **linear**, **rbf** and **laplacian** configurations which I used.

To add a new kernels configuration, append that file with the same format as predefined configs. For example:
```python
'''
Polynomial config
'''
from sklearn.metrics import pairwise # Optional

def polynomial_rgb(X, L=None): # Kernel computing call-back for rgb, always with these input params
	# Return kernel function from package sklearn.metrics.pairwise
	# Add hyperparameters if needed here, in this case, "degree=3"
	return pairwise.polynomial_kernel(X, L, degree=3)

def polynomial_depth(X, L=None): # Kernel computing call-back for depth
	return pairwise.polynomial_kernel(X, L, degree=6)

def polynomial_concatenate(X, L=None): # Kernel computing call-back for concatenated features
	return pairwise.polynomial_kernel(X, L, degree=4)

polynomial_params = Params(name = 'polynomial', # name of the kernels configuration
					assignable_names = ['poly', 'polynomial'], # accepted names when you run command, eg: --kernels=poly
					kernel_func_rgb = polynomial_rgb,
					kernel_func_depth = polynomial_depth,
					kernel_func_concatenate = polynomial_concatenate,
					C_mkl = 0.25, # C of base learner of MKL
					C_rgb = None, # (Optional) C of SVM on rgb, equals C_mkl if None
					C_depth = None, # (Optional) C of SVM on depth, equals C_mkl if None
					C_concatenate = 10, # (Optional) C of SVM on concatenated features, equals C_mkl if None
					lam_mkl = None # (Optional) lamda [0,1] of EasyMKL, 0.0 if None
					)

CONFIGS.append(polynomial_params.to_program_config())
```
### Run
Execute file `run_gesture_classification.py`:
- Run command manually from terminal with the following format:
`python3 run_gesture_classification.py --kinect_train=K3 --kinect_test=K3 --iter_rgb=800 --iter_depth=1600 --kernels=linear`

> Currently supported arguments:
> |Argument|Meaning|eg.|
> |---|---|---|
> |`kinect_train`|Kinect train|K1
> |`kinect_test`|Kinect test|K2
> |`iter_rgb`|Number of iteration of finetuning C3D for extracting RGB features|800
> |`iter_depth`|Number of iteration of finetuning C3D for extracting Depth features|1600
> |`dataset_root`|Overwriting `DATASET_ROOT` set in `configs.py`|/dir
> |`confusion_matrix`|Export confusion matrix or not, default True|True
> |`classification_report`|Export classification report or not, default True|True
> |`kernels`|Kernels configurations set in `configs.py`, accepting only keywords in `assignable_names`|linear


- Run multiple commands sequentially:
-- Modify `evaluation_procedure` file
-- Execute it from terminal `./evaluation_procedure`
> Run `chmod 777 evaluation_procedure` if not working
