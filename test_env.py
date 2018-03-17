print("Start script")
import platform
print(platform.python_version())

print("Importing Modules")

try:
    import tensorflow
    print("TensorFlow version:", tensorflow.__version__)
    import keras
    import h5py
    print("Keras version:", keras.__version__)
    import matplotlib
    import numpy
    import cv2
    print("OpenCV version:", cv2.__version__)
    import nltk
    print("All good to go!!")
except Exception as e:
    print("Error in importing modules!")
    print(str(e))

print("End of script...")
