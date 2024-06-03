# Stress-Detection

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Results](#results)


## Project Overview
Buiilt a stress detection algorithm to classify between stress and not-stressed tasks for 15 subjects. 

The algorithm takes into account physiological and motion data, recorded from both a wrist- and a chest-worn device.

## Usage
The code uses the FLIRT package to calculate features from wearable data such as that from your smartwatch or smart ring: https://flirt.readthedocs.io/en/latest/index.html



## Data
The dataset used in the script is from the University of Siegen under the name WESAD (Multimodal Dataset for Wearable Stress and Affect Detection) : https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

It is a publically available dataset for wearable stress and affect detection that includes the following sensors: blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and three-axis acceleration. 

The dataset studies the connection betweeen stress and emotions, with the 3 classifications being baseline. stress, amusement.



## Results
The University of Siegen achieved classification accuracies of up to 93%. However through my script, I was able to achieve a 97% accuracy with the XGBoost algorithm. 
