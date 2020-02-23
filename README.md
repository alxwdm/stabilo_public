# stabilo_public
Public Repo for the STABILO Ubicomp 2020 Challenge

# About the Challenge

This year, the [UbiComp Conference](http://ubicomp.org/ubicomp2020/) has announced a [Time-Series Classification Challenge](http://ubicomp.org/ubicomp2020/cfp/challenge.html). The competition is about the [STABILO DigiPen](https://www.stabilodigital.com), a sensor-equipped pen that writes on normal paper. The goal is basically about recognizing hand-written letters with the sensor information. 

For me, this multi-staged machine learning challenge is a great way to get more hands-on experience on raw, multi-variate, time-series data. I don't really aim to win the competition (although I wouldn't mind doing so), but instead I want to improve my skills and continue my ML journey. With this in mind, I will not solely aim for the highest score, but I will try to apply some tools that I have learned recently. For example, I want to use an Apache Beam pipeline as one of the preprocessing steps. 

This challenge offers great learning opportunity, covering all ML steps from preprocessing to deployment. In the beginning, I will do most of the work in a private repo, occasionally publishing code snippets as the project progresses. Later, I'll publish all of the code. Stay tuned! 

# Stage 1 - 26 upper case letters

The task in [stage 1](https://stabilodigital.com/data/) is to classify 26 upper case letters. In total, 100 volunteers provided over 13k samples of hand-written letters. 
