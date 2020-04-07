# stabilo_public
Public Repo for the STABILO Ubicomp 2020 Challenge

# About the Challenge

This year, the [UbiComp Conference](http://ubicomp.org/ubicomp2020/) has announced a [Time-Series Classification Challenge](http://ubicomp.org/ubicomp2020/cfp/challenge.html). The competition is about the [STABILO DigiPen](https://www.stabilodigital.com), a sensor-equipped pen that writes on normal paper. The goal is basically about recognizing hand-written letters with the sensor information. 

<p align="center">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/stabilo_digipen.png" width="500">
</p>
  
For me, this multi-staged machine learning challenge is a great way to get more hands-on experience on raw, multi-variate, time-series data. I don't really aim to win the competition, but instead I want to improve my skills and continue my ML journey. With this in mind, I will not solely aim for the highest score, but I will try to apply some cool tools for learning purposes. For example, I will use an Apache beam pipeline for the initial character splitting, although a script with that functionality has already been provided.

This challenge offers great learning opportunity, covering all ML steps from preprocessing to deployment. I will do most of the work in a private repo, occasionally publishing code snippets as the project progresses. Stay tuned! 

# Stage 1 - Classify 26 upper case letters

## Data Preparation with Apache Beam

The task in [stage 1](https://stabilodigital.com/data/) is to classify 26 upper case letters. In total, 100 volunteers provided over 13k samples of hand-written letters. The recordings are saved as csv-files on a per-person basis. The "challenge owner" provided a helper script to split the individual characters from the recording files. I have modified the script and built it into an Apache Beam pipeline. I used Apache Beam primarily for training purposes, but in theory the pipeline scales so that the csv-recordings can be processed in parallel. Within the pipeline, I also separate the data into a train/dev/test set based on a given training ratio. To avoid data leakage, the splitting needs to be done on a per-person basis.

```
p = beam.Pipeline()
filepaths = (p 
    | 'matchfiles' >> beam.io.fileio.MatchFiles(known_args.filepath + '*/sensor_data.csv')
    | 'readmatches' >> beam.io.fileio.ReadMatches()
    | 'getpaths' >> beam.Map(lambda x: (x.metadata.path))
    | 'filterpaths' >> beam.Map(filter_path))
  datasets = filepaths | 'split' >> beam.Map(split_char_and_dataset, known_args.filepath, float(known_args.ratio))
```

Here's a code snippet on how the Apache Beam pipeline is called. The script inputs a directory to the data and a desired training ratio and saves the split characters as individual csv-files into train/dev/test folders.

```
%%bash
python -m split_char_and_sets \
  --filepath "${DATA_PATH}" \
  --ratio "${TRAIN_RATIO}"
```

## Visualization 

There is no machine learning without making friends with the data. It is interessting to see whether there is a visual difference in the sensor data of the same character between different persons, and also how much the individual characters vary. I have looked at several charts from the data set, here are two example plots.

<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/raw_data_gyro.png">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/raw_data_force.png">

The gyroscope data shows a high variance between persons. Also, the signal is quite noisy and different characters are not visually distinguishable. However, the recorded force shows quite a unique temporal pattern for each letter, with varying amplitude between persons. From looking at the data, I guess that the raw force signal has the highest feature importance in an end-to-end approach. Also, it seems that advanced preprocessing and feature engineering will be necessary in order to get useful information out of the other sensor data. 

## Preprocessing pipeline with the tf.data API

After the initial data splitting, each csv-recording contains a single training sample. Now it is time for the actual preprocessing pipeline. I will use the [tf.data API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for this, which is an efficient and scalable way to handle the data. 

The goal of the preprocessing pipeline is to create a tf.data.Dataset and apply mapping functions to it. I used a generator function that reads the files and yields tuples of `(feature, label)` pairs. The feature tensor contains all the data from the csv-files. For the label tensor I chose to encode it as a sparse categorical integer. After applying downstream preprocessing functions (and also depending on the mode), the data is shuffled and batched into `(batch_size, timesteps, n_features)` mini-batches and ready for training. 

Here is how the input function looks like (some lines of code are truncated):

```
def _input_fn()
  def _generator():
    # (...) a generator that reads the csv files and yields (feature, label) tensors
    yield feature, label
  # generate the tf.data.Dataset
  dataset = tf.data.Dataset.from_generator(_generator, output_types=(tf.float32, tf.int32))
  # apply preprocessing functions (such as normalization etc.)
  dataset = dataset.map(preproc_fn_1) \
                   .map(preproc_fn_2)
  # determine the mode, shuffle and repeat only when in training 
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE) \
                     .padded_batch(BATCH_SIZE, padded_shapes=([None,N_FEATURES],1)) \
                     .repeat(TRAIN_STEPS) \
                     .prefetch(PREFETCH_SIZE)
  else:
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=([None,N_FEATURES],1)) \
                     .prefetch(PREFETCH_SIZE)
  return dataset
```

**Improving input pipeline performace:** As mentioned before, each sample is stored in a separate csv-file after the initial character splitting. In a professional cloud environment, this may be the typical use case for ETL-pipelines. However, storing the data on Google Drive and reading it into Colab does not scale. In fact, reading the data turned out to be a bottleneck in my training workflow. This is why I have transformed the dataset into a binary tfrecord-file. You can read about TFRecord in [this TensorFlow guide](https://www.tensorflow.org/tutorials/load_data/tfrecord). To do so, the tensors from the original dataset must be converted to a serialized string using tf.Example. Using the existing (csv-based) dataset from above, it looks something like this:

```
def serialize_example(elem):
  # extract features and labels from dataset element and serialize tensors
  features = elem[0]
  labels = elem[1]
  feature_string = tf.io.serialize_tensor(features)
  label_string = tf.io.serialize_tensor(labels)
  # turn data into tf.Example message
  data = {'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_string.numpy()])),
          'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string.numpy()]))}
  example_proto = tf.train.Example(features=tf.train.Features(feature=data))
  # return serialized tf.Example
  return example_proto.SerializeToString()

# helper generator function that yields serialized examples
def _generator():
  for elem in dataset:
    yield serialize_example(elem)

# create serialized dataset from helper generator 
serialized_dataset = tf.data.Dataset.from_generator(_generator, output_types=tf.string, output_shapes=())

# save the data as *.tfrecord file
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_dataset)
```

Changing from csv to tfrecord improved the training speed at least by a factor of 10. There are further ways to optimize input pipelines with the tf.data API, as it can be [read in this guide](https://www.tensorflow.org/guide/data_performance). The following picture (inpired by the guide linked above) demonstrates how things like prefetching, parallel processing and so on can increase the GPU usage - and thus speed up training - when reading the data is the bottleneck:

<p align="center">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/data_pipeline.png" width="600">
</p>

## Model workflow and debugging with the tf.estimator API

I decided to use the [tf.estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator) as a framework for the modelling workflow. This is a very powerful and highly optimized API which is capable of both local and distributed multi-server training without having to change the code. One of the core principles of the tf.estimator API is to separate the data pipeline from the model. Also, the checkpointing and logging is done for you and ready to be visualized with TensorBoard. I will not mention more details, but instead I provide a link to the official [tf.estimator Guide](https://www.tensorflow.org/guide/estimator) and a link to a [comprehensive article on tds](https://towardsdatascience.com/an-advanced-example-of-tensorflow-estimators-part-1-3-c9ffba3bff03) about the framework.

**Unit testing and sanity checks:** Currently, the integration of custom keras models into the tf.estimator API is not so seamless as it appears on a first glance. It took me an enourmous amount of effort and time to get everything to work, diving deep into TensorFlow.  Finally, the estimator passed the testing and sanity checks I applied for debugging. For example, I had to modify the loss function so I checked whether the initial sparse cross entropy loss is around the expected value of `-ln(1/N_CLASSES)`. Also, I tested whether the model is able to overfit to a single training sample in order to verify the training workflow with `tf.GradientTape()`.

**Debugging of Training, Serving and Deployment:** In order to iterate more quickly, I have reduced the classification task down to just the letters A, B and C. This makes debugging the rest of the workflow (i.e. training on more data, serving and deployment) faster. After training for a few epochs and without a thorough model architecture or hyperparameter search, I was able to reach an accuracy close to 97% on the dev set. 

When it comes to serving, the use of the tf.estimator API pays off. Following [this guide](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators), writing a prediction function was fairly easy. The function takes a path to a csv-file as input and converts the data into a feature tensor. Then, the model is loaded and the features are passed to the `serving_input_receiver_fn` of the estimator, where the preprocessing takes place. Finally, the predicted letter and corresponding probability of the softmax output is printed out. For batch prediction of multiple csv-files, the model is only loaded once.

As you can see below, a sample from the dev set is correctly classified with high accuracy:
```
FILE_PATH = DATA_PATH + 'dev_reduced/15_1_A.csv'
prediction = predict(FILE_PATH)
Predicted character A with a probability of 99.99%.
```

For deployment, the model needs to be converted into an executable that runs in a Windows 10 command window. After passing a directory string to the csv-files and to the calibration file, the results need to be printed in a certain format (see [submission details](https://stabilodigital.com/submissions/)). Here is a sample output from the test set - person number 36 - with the model trained on the reduced data. The file name indicates the correct label (this will not be the case for the challenge validation files, so don't think about cheating) and the letter printed after the three stars is the prediction for that file. For the reduced classification task, the predictions are accurate!
```
/STABILO/challenge1_data/test_reduced/36_1_A.csv***A~~~/STABILO/challenge1_data/test_reduced/36_3_C.csv***C~~~
/STABILO/challenge1_data/test_reduced/36_27_A.csv***A~~~/STABILO/challenge1_data/test_reduced/36_2_B.csv***B~~~
/STABILO/challenge1_data/test_reduced/36_29_C.csv***C~~~/STABILO/challenge1_data/test_reduced/36_28_B.csv***B~~~
```

## Training on the whole dataset and Error Analysis

Finally, I have trained on the complete dataset to classify all 26 upper-case characters. I used a model with multiple bidirectional LSTM and Dense layers with about 500k parameters in total. Training was done with SGD optimizer. Because of quota limits, I could not perform an intense hyperparameter search, but at least I was able to tune the learning rate and regularization parameters in order to reduce the overfitting that occured with the initial model. The following picture shows the learning curves from TensorBoard before and after the tuning. The smoothing parameter is set equally, the larger noise in the right is due to the higher dropout rate. 

<p align="center">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/param_tuning.png" width="600">
</p>

**Error Analysis:** Without fine-tuning, the accuracy on the dev set gets close to 80% after about 50 epochs, which is not too bad for 26 classes (random guessing yields about 4% accuracy). Plotting the confusion matrix gives interessting insights:

<p align="center">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/confusion_matrix.png" width="400">
</p>

Most of the characters are classified correctly with high accuracy. However, some characters appear to be particularly difficult to recognize. Besides the "usual suspects" from image-based classification tasks such as `(I, J)`, it turned out that characters with similar temporal force patterns are confused by the model - as already suspected in the visualization section above. For example, `P` almost always gets classified as `D`. Other characters with similar force patterns are `(O, D)`, `(X, T)` and `(L, C)`. In order to distinguish those pairs with higher accuracy, I guess it would be necessary to apply further preprocessing to the input signals. However, training for a longer time and fine-tuning the hyperparameters might also boost up the accuracy.
