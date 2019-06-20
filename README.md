# kaggle-imet
https://www.kaggle.com/c/imet-2019-fgvc6

# In a nutshell

The classifier pipleline based on IMet CVPR challenge hosted on kaggle. 
For details see:
https://www.kaggle.com/c/imet-2019-fgvc6 
In this challenge one needs to classify the items in The Metropolitan Museum of Art in New York.  
The exponates have general "cultures" categories and more concrete "tag" attributes giving 1103 classes in total.

# Download dataset
Competition dataset can be downloaded here: 
https://www.kaggle.com/c/imet-2019-fgvc6/data
For faster download it's advasable to install kaggle API and use command line:
kaggle competitions download -c imet-2019-fgvc6

# Make folds
Train csv with folds provided can be generated using make-folds.py
It gives stratified folds using this package: https://github.com/trent-b/iterative-stratification
Or using this approach from Kontantin Lopuhin: https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/make_folds.py

# Run training
Download competition dataset. 
Set data root and model saving run root in configs.py
Generate folds.
Call python train.py
Note: training requires time with one epoch taking approximately 50 min on P100 GPU. 

# Run prediction
Download pre-trained model weights here: https://www.kaggle.com/blondinka/seresnext101-folds
Call python inference.py

For all-in-one inference kernel see:
https://www.kaggle.com/blondinka/predict-submit-seresnext101

Have fun!
