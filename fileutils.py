import os

def recursive_walk(rootdir, concaterootdir = True):
    if concaterootdir:
        for r, dirs, files in os.walk(rootdir):
            for f in files:
                yield os.path.join(r, f)
    else:
        for r, dirs, files in os.walk(rootdir):
            for f in files:
                yield f

def listdir(rootdir, concaterootdir = True):
	dirs = os.listdir(rootdir)
	if concaterootdir:
		for i in range (0, len(dirs)):
			dirs[i] = os.path.join(rootdir, dirs[i])
	return dirs

def dirname(path):
	if('/' in path):
		return path[path.rfind('/') + 1:]
	return path

def filename(path):
	if('/' in path):
		return path[path.rfind('/') + 1:path.rfind('.')]
	return path

def fileextension(path):
	return path[path.rfind('.') + 1:]

def assureDir(pathToDir):
	if not os.path.exists(pathToDir):
		os.makedirs(pathToDir)
