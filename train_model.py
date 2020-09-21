from StomataDetector.Detection.Custom import DetectionModelTrainer
import os
import tensorflow as tf
# from tensorflow.python.client import device_lib
"""
    Training for models
"""
# print(device_lib.list_local_devices()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="cells")
trainer.setTrainConfig(object_names_array=["stomata"],
                       batch_size=8, num_experiments=100,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()





