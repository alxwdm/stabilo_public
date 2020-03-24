# stabilo_public
Public Repo for the STABILO Ubicomp 2020 Challenge

# About the Challenge

This year, the [UbiComp Conference](http://ubicomp.org/ubicomp2020/) has announced a [Time-Series Classification Challenge](http://ubicomp.org/ubicomp2020/cfp/challenge.html). The competition is about the [STABILO DigiPen](https://www.stabilodigital.com), a sensor-equipped pen that writes on normal paper. The goal is basically about recognizing hand-written letters with the sensor information. 

<p align="center">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/stabilo_digipen.png" width="500">
</p>
  
For me, this multi-staged machine learning challenge is a great way to get more hands-on experience on raw, multi-variate, time-series data. I don't really aim to win the competition (although I wouldn't mind doing so), but instead I want to improve my skills and continue my ML journey. With this in mind, I will not solely aim for the highest score, but I will try to apply some tools that I have learned recently. For example, I want to use an Apache Beam pipeline as one of the preprocessing steps. 

This challenge offers great learning opportunity, covering all ML steps from preprocessing to deployment. In the beginning, I will do most of the work in a private repo, occasionally publishing code snippets as the project progresses. Later, I'll publish all of the code. Stay tuned! 

# Stage 1 - 26 upper case letters

**About the Data:** The task in [stage 1](https://stabilodigital.com/data/) is to classify 26 upper case letters. In total, 100 volunteers provided over 13k samples of hand-written letters. The recordings are saved as csv-files on a per-person basis. The "challenge owner" provided a helper script to split the individual characters from the recording files. I have modified the script and built it into an Apache Beam pipeline. I used Apache Beam primarily for training purposes, but in theory the pipeline scales so that the csv-recordings can be processed in parallel. Within the pipeline, I also separate the data into a train/dev/test set based on a given training ratio. To avoid data leakage, the splitting needs to be done on a per-person basis.

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

**Visualization:** There is no machine learning without making friends with the data. It is interessting to see whether there is a visual difference in the sensor data of the same character between different persons, and also how much the individual characters vary.

<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/raw_data_gyro.png">
<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/raw_data_force.png">

The gyroscope data shows a high variance between persons and different characters are not visually distinguishable. However, the recorded force shows quite a unique temporal pattern for each letter, with varying amplitude between persons. 

**Preprocessing with the tf.data API:** After the initial data splitting, each csv-recording contains a single training sample. Now it is time for the actual preprocessing pipeline. I will use the [tf.data API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for this, which is an efficient and scalable way to handle the data. 

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

**Model workflow with the tf.estimator API:** I decided to use the [tf.estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator) as a framework for the modelling workflow. This is a very powerful and highly optimized API which is capable of both local and distributed multi-server training without having to change the code. One of the core principles of the tf.estimator API is to separate the data pipeline from the model. Also, the checkpointing and logging is done for you and ready to be visualized with TensorBoard. I provide a link to the official [tf.estimator Guide](https://www.tensorflow.org/guide/estimator) and a link to a [comprehensive article on tds](https://towardsdatascience.com/an-advanced-example-of-tensorflow-estimators-part-1-3-c9ffba3bff03) about the framework.

**Unit testing and sanity checks:** Currently, the integration of **custom keras models into the tf.estimator API** is not so seamless as it appears on a first glance. It took me an enourmous amount of effort and time to get everything to work. This was indeed a very bumpy road, but finally the estimator passed the testing and sanity checks I applied for debugging. For example, I checked whether the initial sparse cross entropy loss around the expected value of `-ln(1/N_CLASSES)`. Also, I tested whether the model is able to overfit to a single training sample.

Here is the prediction output after training on one sample for a few iterations:
```
Debugging model...
Predicted character (index): A (0)
Predicted probabilities: [9.9965453e-01 2.4235590e-06 ... 1.0922228e-05]
Loss for debug sample: 0.0003456472
```
