import pathlib as p
import tensorflow as tf
import numpy as np
path = p.Path.cwd()

config = {
    "H": -1,
    "W": -1,
    "num_channels": 3,
    "batch_size_per_gpu": 1,
    "adding_noise": True,
    "adding_blur": False,  # True if the model is uploaded
    "adding_compression": False,  # True if the model is uploaded
    "test_stdVec": [5.],  # noise level (sigma/255)
    "test_blurVec": [0],
    "test_compresVec": [1.],
    "localhost": None,
    "img_types": ["png", "jpg", "jpeg", "bmp", "dcm"],
    "num_sel_imgs": -1
}

import DataLoader_colab
# exec(open('DataLoader_colab.py').read())
# %run '/content/ADL/TensorFlow/util/DataLoader_colab.py


class Denoiser:

    def __init__(self):
        self.path = p.Path.cwd()
        self.input_dir = f'{self.path}/input'
        self.output_dir = f'{self.path}/output'
        self.gt_ds, self.inp_ds, self.adl_results, self.noise_level = [], [], [], []

    def run_model(self):
        self.model = tf.keras.models.load_model(
            f'{path}/TensorFlow/pretrained_models/RGB_model_WGN/checkpoint-36/',
            compile=False)
        # model.load_weights(weights_path, by_name=True,
        #                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    def run_colab(self, file_format, conv_func, image_name):
        test_ds = DataLoader_colab.DataLoader(config=config, test_ds_dir=self.input_dir,
                                              file_format=file_format,
                                              conv_func=conv_func,
                                              image_name=image_name)()

        for ds_name, DSs in test_ds.items():
            print(f"dataset: {ds_name}...")

            for distortion_name, DS in DSs.items():
                print(f"\tdistortion type: {distortion_name}...")
                sigma = int(float(distortion_name.split('_wgn_')[-1]))
                self.noise_level.append(sigma)

                for inp, gt, img_name in DS.batch(1):
                    y_hat, _, _ = self.model.predict(inp)

                    self.gt_ds.append(np.squeeze(gt.numpy()).astype(np.float32))
                    self.inp_ds.append(np.squeeze(inp.numpy()).astype(np.float32))
                    self.adl_results.append(np.squeeze(tf.identity(y_hat).numpy()).astype(np.float32))

        return self.inp_ds, self.adl_results
