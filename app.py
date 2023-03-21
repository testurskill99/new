import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from flask_cors import CORS
import pathlib as p
import denoising
import tensorflow as tf
import numpy as np
from math import log10, sqrt
from datetime import datetime
plt.switch_backend('agg')

path = p.Path.cwd()
input_dir = f'{path}/input'
output_dir = f'{path}/output'

app = Flask(__name__, template_folder=f'{path}/build', static_folder=f'{path}/build/static')
CORS(app)


def make_directory(f1_format, f2_format):
    if not p.Path(output_dir).exists():
        p.Path(output_dir).mkdir()
        p.Path(f'{output_dir}/{f1_format}_to_{f2_format}').mkdir()
    elif p.Path(f'{output_dir}/{f1_format}_to_{f2_format}').exists():
        pass
    else:
        p.Path(f'{output_dir}/{f1_format}_to_{f2_format}').mkdir()


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_extension = f.filename.split('.')[-1]
        f.save(f'input/{file_extension}/{f.filename}')

        op = {'status_code': 200, 'data': 'Successfully uploaded', 'error': 'null'}

        return op


@app.route('/denoise/pngtopng/<image_name>', methods=['GET', 'POST'])
def png_to_png_execution(image_name):

    conv_path = 'png_to_png'
    input_file_format = 'png'
    output_file_format = 'png'

    inp_ds, adl_results = master_func(conv_path, input_file_format, output_file_format, image_name)
    filename = display(-1, inp_ds, adl_results, f'{output_dir}/{conv_path}', output_file_format, conv_path)
    # psnr_val = psnr(f'{output_dir}/{conv_path}', output_file_format)

    return send_file(filename, as_attachment=True)


@app.route('/denoise/dcmtopng/<image_name>', methods=['GET', 'POST'])
def dcm_to_png_execution(image_name):

    conv_path = 'dcm_to_png'
    input_file_format = 'dcm'
    output_file_format = 'png'

    inp_ds, adl_results = master_func(conv_path, input_file_format, output_file_format, image_name)
    filename = display(-1, inp_ds, adl_results, f'{output_dir}/{conv_path}', output_file_format, conv_path)
    # psnr_val = psnr(f'{output_dir}/{conv_path}', output_file_format)

    return send_file(filename, as_attachment=True)


@app.route('/denoise/dcmtojpeg/<image_name>', methods=['GET', 'POST'])
def dcm_to_jpeg_execution(image_name):

    conv_path = 'dcm_to_jpeg'
    input_file_format = 'dcm'
    output_file_format = 'jpeg'

    inp_ds, adl_results = master_func(conv_path, input_file_format, output_file_format, image_name)
    filename = display(-1, inp_ds, adl_results, f'{output_dir}/{conv_path}', output_file_format, conv_path)
    # psnr_val = psnr(f'{output_dir}/{conv_path}', output_file_format)

    return send_file(filename, as_attachment=True)


def master_func(conv_path, input_file_format, output_file_format, image_name):

    make_directory(input_file_format, output_file_format)
    obj1 = denoising.Denoiser()
    obj1.run_model()
    inp_ds, adl_results = obj1.run_colab(input_file_format, conv_path, image_name)

    return inp_ds, adl_results


def display(img_id, inp_ds, adl_results, output_path, output_file_format, conv_path):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    denoised_op_path = f'output/{conv_path}/denoised_{dt_string}.{output_file_format}'

    # # saving noisy image
    # plt.imshow(inp_ds[img_id])
    # plt.savefig(f'{output_path}/noisy_image_{dt_string}.{output_file_format}')
    # plt.close()

    # # saving denoised image
    plt.imshow(adl_results[img_id])
    plt.axis('off')
    plt.savefig(f'{output_path}/denoised_{dt_string}.{output_file_format}',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    return denoised_op_path


def psnr(img_path, output_file_format):

    image = tf.io.read_file(
        f'{img_path}\\noisy_image_.{output_file_format}')
    img1 = tf.io.decode_jpeg(image, channels=3)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    image = tf.io.read_file(
        f'{img_path}\\denoised_.{output_file_format}')
    img2 = tf.io.decode_jpeg(image, channels=3)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr_val = 20 * log10(max_pixel / sqrt(mse))

    return f"PSNR value between noisy and denoised image is {psnr_val} dB"


def display_comparison(img_id, inp_ds, adl_results, output_path, output_file_format, psnr_val):
    # # saving comparison image
    num_figs = 2
    fontsize = 15

    fig = plt.figure(figsize=(num_figs * 6, 8))

    ax2 = fig.add_subplot(1, num_figs, 1)
    plt.title(f'Noisy Image', fontsize=fontsize)
    ax2.axis('off')
    ax2.imshow(np.clip(inp_ds[img_id], 0, 1))

    ax3 = fig.add_subplot(1, num_figs, 2)
    plt.title('ADL output', fontsize=fontsize)
    ax3.axis('off')
    ax3.imshow(adl_results[img_id])
    fig.suptitle(psnr_val, fontsize=20)

    plt.savefig(f'{output_path}/Comparison_.{output_file_format}')
    plt.close()


if __name__ == "__main__":
    app.run(debug=True)
