# MAIS202
Uses the following dataset: https://www.kaggle.com/datasets/nwheeler443/gono-unitigs/data. <br> <br>
After obtaining the sequencing data of the bacterial strain a patient is afflicted by, the health care practitioner could paste it into the web app, and it will determine if the strain has resistance against Azythromycin, Cefixime, and/or Ciprofloxacin. The practitioner can then prescribe the appropriate medication.
<br> <br>
Final project of the MAIS202 Bootcamp. 

# Web App
![image](https://github.com/angelaxzhu/MAIS202/assets/125671211/381a98e8-710b-4d6b-b0a4-5f0eb4359c0a)
<br> Built with Flask. 

# Training Results
In general, there was better accuracy than precision and recall. In other words, the model is better at only labelling a sample as resistant when it is truly resistant, but less good at finding all resistant samples. 
## AZM
![image](https://github.com/angelaxzhu/MAIS202/assets/125671211/7df84e07-07a7-4c45-bf95-9cd7ed4a41d4)

- Accuracy: 0.960960960960961
- Precision: 0.8740157480314961 
- Recall: 0.7900355871886121 

## Cip
![image](https://github.com/angelaxzhu/MAIS202/assets/125671211/ab811992-5b00-409b-b906-6301d6496ea1)

- Accuracy: 0.9598840019333011
- Precision: 0.9578833693304536
- Recall: 0.9527389903329753 

## Cfx
![image](https://github.com/angelaxzhu/MAIS202/assets/125671211/09ab7a06-a88f-4a9d-a18b-a652bc2e4509)

- Accuracy: 0.9960508995173322
- Precision: 0.0
- Recall: 0.0
<br>
Note: For cefixime, the dataset only five samples that are cefixime resistant, when splitting the dataset, it's possible that none were sorted in the testing set, which is why there are no true positives in the confusion matrix.

# Future Works
- Change the colour of text if it's resistant/ non-resistant
- Include other factors like geography
- Extend to other drugs => find a way to find appropriate unitigs 
