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

**Preprocessing:** After the initial data splitting, each csv-recording contains a single training sample. Now it is time for the actual preprocessing pipeline. I will use the tf.data API for this, which is an efficient and scalable way to handle the data. 

Here are a few datapoints of a single recording from the tf.data.Dataset after reading in the csv files. The example below does not show any downstream preprocessing functions applied to the data:

```
dataset = tf.data.Dataset(...) # some code to create the dataset

for batch, label in dataset.take(1):
  print("{:10s}: {}".format("Label",label[0]))
  for key, value in batch.items():
    print("{:10s}: {}".format(key,value.numpy()[0:9]))

Label     : b'S'
Acc1 X    : [-7878. -8023. -7892. -7862. -7673. -8003. -7886. -7856. -7804.]
Acc1 Y    : [12740. 13243. 12917. 12900. 12783. 13101. 12709. 12711. 12697.]
Acc1 Z    : [-6597. -6755. -6748. -6474. -6396. -6395. -6250. -6212. -6297.]
Acc2 X    : [-1938. -1952. -1719. -1674. -1741. -1967. -1978. -1937. -2048.]
Acc2 Y    : [-3119. -3259. -3197. -3195. -3171. -3171. -3212. -3142. -3134.]
Acc2 Z    : [1509. 1608. 1629. 1683. 1646. 1616. 1508. 1527. 1601.]
Gyro X    : [-76. -41. -21. -13. -25. -30.  -9.  12.   2.]
Gyro Y    : [ 85. 115.  31.  13. -34.   1. -41.  -9.   6.]
Gyro Z    : [133. 141. 111.  33. -18. -11.   1.  -4.  35.]
Mag X     : [-74. -75. -75. -76. -76. -74. -77. -75. -77.]
Mag Y     : [142. 141. 142. 143. 143. 140. 142. 138. 140.]
Mag Z     : [263. 263. 261. 261. 260. 259. 261. 259. 259.]
Force     : [1. 1. 2. 5. 8. 7. 7. 5. 5.]
```
