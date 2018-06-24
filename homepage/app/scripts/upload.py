################### LOAD PACKAGES ###################
from flask import send_from_directory, send_file, render_template
from werkzeug import secure_filename
import os
import sys
from scripts.skin_detection import crop_img
from PIL import Image

import tensorflow as tf
import numpy as np
from datetime import datetime
import random
import argparse
import regex as re

data_generator = tf.keras.preprocessing.image.ImageDataGenerator
pool = tf.keras.layers.MaxPooling2D
conv = tf.keras.layers.Convolution2D
dense = tf.keras.layers.Dense
relu = tf.keras.activations.relu
flatten = tf.keras.layers.Flatten
dropout = tf.keras.layers.Dropout
glob_avg_pool = tf.keras.layers.GlobalAvgPool2D
preprocess_input = tf.keras.applications.vgg16.preprocess_input
VGG16 = tf.keras.applications.vgg16.VGG16
regularizer = tf.keras.regularizers.l2(l = 0.0005) # scale inspired by VGG16-paper

################### HELPER FUNCTIONS ###################

def generate_mask_from_task(task):

    mask0 = [1, 0, 0]
    mask1 = [0, 1, 0]
    mask2 = [0, 0, 1]

    if task == 1:
        return(np.array(mask0))
    elif task == 2:
        return(np.array(mask1))
    else:
        return(np.array(mask2))

def masked_sigmoid_cross_entropy_with_logits(logits, labels, masks):
    return tf.multiply(masks, tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['jpg','bmp','gif','jpeg','png'])

def compute_certainty(sigmoid_output, prediction):
    if prediction == 0:
        return 1.0 - sigmoid_output
    else:
        return sigmoid_output


################### MODEL ###################

def vgg16_model(x, m, PT_model):
    with tf.variable_scope('network'):
        PT_layer = PT_model(x)

        x_gap = glob_avg_pool(name='glob_avg_pool')(PT_layer)

        # # %% flatten
        x_f1 = flatten(name='flatten')(x_gap)
        x_de = dense(64, activation='relu', name='fc1', kernel_regularizer = regularizer)(x_f1)

        # %% IN CASE OF PSORIASIS OR ECZEMA
        x_de0 = dense(32, activation='relu', name='fc01', kernel_regularizer = regularizer)(x_de)
        y_logits0 = dense(1, name='predictions0')(x_de0)

        # %% IN CASE OF ACNE VULGARIS OR ROSACEA
        x_de1 = dense(32, activation='relu', name='fc11', kernel_regularizer = regularizer)(x_de)
        y_logits1 = dense(1, name='predictions1')(x_de1)

        # %% IN CASE OF ECZEMA OR MYCOSIS FUNGOIDES
        x_de2 = dense(32, activation='relu', name='fc21', kernel_regularizer = regularizer)(x_de)
        y_logits2 = dense(1, name='predictions2')(x_de2)

        y_logits = tf.concat([y_logits0, y_logits1, y_logits2], 1)

        n_samples = tf.reduce_sum(tf.count_nonzero(m, axis=1, dtype=tf.float32))

    with tf.variable_scope('performance'):
        probabilities = tf.multiply(tf.sigmoid(y_logits), m)
        prediction = tf.round(probabilities)

    with tf.variable_scope('tasks'):
        t0 = tf.constant([1, 0, 0], dtype=tf.float32)   # psoriasis and eczema
        t1 = tf.constant([0, 1, 0], dtype=tf.float32)   # acne and rosacea
        t2 = tf.constant([0, 0, 1], dtype=tf.float32)   # mycosis and eczema
       
        b0 = tf.reduce_all(tf.equal(m, t0), axis=1)     # psoriasis and eczema
        b1 = tf.reduce_all(tf.equal(m, t1), axis=1)     # acne and rosacea
        b2 = tf.reduce_all(tf.equal(m, t2), axis=1)     # mycosis and eczema

        i0 = tf.where(b0)  # psoriasis and eczema
        i1 = tf.where(b1)  # acne and rosacea
        i2 = tf.where(b2)  # mycosis and eczema

        p0 = tf.gather(tf.slice(prediction, [0, 0], [-1, 1]), i0) # collect task 0 predictions
        p1 = tf.gather(tf.slice(prediction, [0, 1], [-1, 1]), i1) # collect task 1 predictions
        p2 = tf.gather(tf.slice(prediction, [0, 2], [-1, 1]), i2) # collect task 2 predictions

        probabilities_t0 = tf.gather(tf.slice(probabilities, [0, 0], [-1, 1]), i0)
        probabilities_t1 = tf.gather(tf.slice(probabilities, [0, 1], [-1, 1]), i1)
        probabilities_t2 = tf.gather(tf.slice(probabilities, [0, 2], [-1, 1]), i2)

        return p0, p1, p2, probabilities_t0, probabilities_t1, probabilities_t2

def handle_file(request, folder):
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']

    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return 'No file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder, filename)
        file.save(filepath)

        img = crop_img(filepath)
        img_crop = Image.fromarray(img)
        img_crop.save(filepath + "_crop.jpg")
        width, height = img_crop.size
        
        if width > 4000 or height > 4000:
            res_width = int(width/4.0)
            res_height = int(height/4.0)
        elif width > 3000 or height > 3000:
            res_width = int(width/3.0)
            res_height = int(height/3.0)
        elif width > 2000 or height > 2000:
            res_width = int(width/2.0)
            res_height = int(height/2.0)
        else:
            res_height = height
            res_width = width 

        min_side = min(res_height, res_width)

        res_height = min_side
        res_width = min_side      

        image_res = img_crop.resize((res_width,res_height))
        img = np.array(image_res)

        height, width, n_channels = img.shape

        task = int(request.form.get("tasks", None))

        skin_diseases0 = ["Psoriasis", "Eczema"]
        skin_diseases1 = ["Acne Vulgrais", "Rosacea"]
        skin_diseases2 = ["Mycosis Fungoides", "Eczema"]

        tf.reset_default_graph()

        PT_model = VGG16(weights = "imagenet",
         include_top = False,
         input_tensor = None,
         input_shape = None,
         pooling = None)

        PT_model.layers.pop()

        for i, layer in enumerate(PT_model.layers):
            if i < 15:
                layer.trainable = False

        x = tf.placeholder(tf.float32, [None, height, width, n_channels], name="x_pl")
        m = tf.placeholder(tf.float32, [None, 3], name="m_pl")

        pd0, pd1, pd2, pb0, pb1, pb2 = vgg16_model(x, m, PT_model)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./model/vgg16_model.ckpt")

            img = img.reshape((1,height,width,n_channels))
            img = preprocess_input(img.astype(np.float32))
            mask = generate_mask_from_task(task)
            mask = mask.reshape((1,3))
            feed = {x: img, m: mask}

            pred0, pred1, pred2, prob0, prob1, prob2 = sess.run([pd0, pd1, pd2, pb0, pb1, pb2], feed_dict = feed)

            if task == 1:
                pred = skin_diseases0[int(pred0[0])]
                certainty = compute_certainty(float(prob0[0][0]), int(pred0[0]))
                certainty = "%.2f" % (100*certainty)
            elif task == 2:
                pred = skin_diseases1[int(pred1[0])]
                certainty = compute_certainty(float(prob1[0][0]), int(pred1[0]))
                certainty = "%.2f" % (100*certainty)
            else:
                pred = skin_diseases2[int(pred2[0])]
                certainty = compute_certainty(float(prob2[0][0]), int(pred2[0]))
                certainty = "%.2f" % (100*certainty)

        tf.keras.backend.clear_session()

        return render_template('home.html', filename = filename, filename_zoom = str(filename)+"_crop.jpg", pred = pred, certainty = certainty, task=task)
    else:
        return 'Error', 400

def show_file(filename, upload_folder):
    return send_from_directory(filename, upload_folder)
