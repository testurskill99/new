import pathlib as p
import tensorflow as tf
import numpy as np
import pydicom

path = p.Path.cwd()
output_dir = f'{path}/output'
input_dir = f'{path}/input'
path = p.Path.cwd()


def png_to_png(image_name):
    """
        1. creates output/png_to_png directory
        2. Fetches png image from input/png and decodes the image as numpy array
    """
    if not p.Path(output_dir).exists():
        p.Path(output_dir).mkdir()
        p.Path(f'{output_dir}/png_to_png').mkdir()
    elif not p.Path(f'{output_dir}/png_to_png').exists():
        p.Path(f'{output_dir}/png_to_png').mkdir()
    else:
        pass

    file_path = f'{path}/input/png/{image_name}.png'
    file = tf.io.read_file(file_path)
    img = tf.image.decode_png(file, dtype=tf.uint8, channels=3)

    return img


def dcm_to_png(image_name):
    """
        1. creates output/dcm_to_png directory
        2. Fetches dcm image from input/dcm and decodes the image as numpy array
    """
    if not p.Path(output_dir).exists():
        p.Path(output_dir).mkdir()
        p.Path(f'{output_dir}/dcm_to_png').mkdir()
    elif not p.Path(f'{output_dir}/dcm_to_png').exists():
        p.Path(f'{output_dir}/dcm_to_png').mkdir()
    else:
        pass

    img = dicom_decoder(image_name)

    return img


def dcm_to_jpeg(image_name):
    """
        1. creates output/dcm_to_jpeg directory
        2. Fetches dcm image from input/dcm and decodes the image as numpy array
    """
    if not p.Path(output_dir).exists():
        p.Path(output_dir).mkdir()
        p.Path(f'{output_dir}/dcm_to_jpeg').mkdir()
    elif not p.Path(f'{output_dir}/dcm_to_jpeg').exists():
        p.Path(f'{output_dir}/dcm_to_jpeg').mkdir()
    else:
        pass

    img = dicom_decoder(image_name)

    return img


def dicom_decoder(image_name):
    """
        Fetches dcm image and decodes the image as numpy array
    """
    ds = pydicom.dcmread(f'{path}/input/dcm/{image_name}.dcm')
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 256
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    stacked_img = np.stack((image_2d_scaled,) * 3, axis=-1)
    img = tf.convert_to_tensor(stacked_img)

    return img
