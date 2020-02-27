# stabilo_public
Public Repo for the STABILO Ubicomp 2020 Challenge

# About the Challenge

This year, the [UbiComp Conference](http://ubicomp.org/ubicomp2020/) has announced a [Time-Series Classification Challenge](http://ubicomp.org/ubicomp2020/cfp/challenge.html). The competition is about the [STABILO DigiPen](https://www.stabilodigital.com), a sensor-equipped pen that writes on normal paper. The goal is basically about recognizing hand-written letters with the sensor information. 

<img src="https://github.com/alxwdm/stabilo_public/blob/master/pics/stabilo_digipen.png" width="500">

For me, this multi-staged machine learning challenge is a great way to get more hands-on experience on raw, multi-variate, time-series data. I don't really aim to win the competition (although I wouldn't mind doing so), but instead I want to improve my skills and continue my ML journey. With this in mind, I will not solely aim for the highest score, but I will try to apply some tools that I have learned recently. For example, I want to use an Apache Beam pipeline as one of the preprocessing steps. 

This challenge offers great learning opportunity, covering all ML steps from preprocessing to deployment. In the beginning, I will do most of the work in a private repo, occasionally publishing code snippets as the project progresses. Later, I'll publish all of the code. Stay tuned! 

# Stage 1 - 26 upper case letters

The task in [stage 1](https://stabilodigital.com/data/) is to classify 26 upper case letters. In total, 100 volunteers provided over 13k samples of hand-written letters. The recordings are saved as csv-files on a per-person basis. The "challenge owner" provided a helper script to split the individual characters from the recording files. I have modified the script and built it into an Apache Beam pipeline. I used Apache Beam primarily for training purposes, but in theory the pipeline scales so that the csv-recordings can be processed in parallel. Within the pipeline, I also separate the data into a train/dev/test set based on a given training ratio. To avoid data leakage, the splitting needs to be done on a per-person basis.


Here's a code snippet on how the Apache Beam pipeline is called. The script inputs a directory to the data and a desired training ratio and saves the split characters as individual csv-files into train/dev/test folders.

```
%%bash
python -m split_char_and_sets \
  --filepath "${DATA_PATH}" \
  --ratio "${TRAIN_RATIO}"
```
