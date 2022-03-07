<h1 align="center">Realtime Face Anti-Spoofing Detection :robot:</h1>

![](https://komarev.com/ghpvc/?username=Prem95&style=flat-square&label=Views)
![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)
[![Forks](https://img.shields.io/github/forks/Prem95/face-liveness-detector.svg?logo=github)](https://github.com/Prem95/face-liveness-detector/network/members)
[![Stargazers](https://img.shields.io/github/stars/Prem95/face-liveness-detector.svg?logo=github)](https://github.com/Prem95/face-liveness-detector/stargazers)
[![contributions welcome](https://img.shields.io/badge/Contributions-Welcome-green.svg?style=flat)](https://github.com/Prem95/face-liveness-detector/issues)

<div align= "center"><img src="https://github.com/Prem95/face-liveness-detector/blob/main/misc/demo.gif" width="600" height="550"/></div>


<h3> Contact premstroke95@gmail.com to request the sample model file and utility folder</h3>

## Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [1.0] - 28/12/2021
- First commit


## Why Build This? :thinking:
Face anti-spoofing systems has lately attracted increasing attention due to its important role in securing face recognition systems from fraudulent attacks. This project aims to provide a starting point in recognising real and fake faces based on a model that is trained with publicly available dataset


## Where to use? :hammer:
This Face Anti Spoofing detector can be used in many different systems that needs realtime facial recognition with facial landmarks. Potentially could be used in security systems, biometrics, attendence systems and etc.

Can be integrated with hardware systems for application in offices, schools, and public places for various use cases.

## Datasets and Library :green_book:

The model is trained using Tensorflow from publicly available datasets. Below listed are the data sources that the model is trained on:

CASIA: https://github.com/namtpham/casia2groundtruth

OULU: https://sites.google.com/site/oulunpudatabase/

Please obtain the necessary permissions before using the datasets as above.

## Prerequisites :umbrella:

All the required libraries are included in the file ```requirements.txt```. Tested on Ubuntu 20.04 with Python3.8.

## Installation :computer:
1. Clone the repo
```
$ git clone https://github.com/Prem95/face-liveness-detector.git
```

2. Change your directory to the cloned repo
```
$ cd face-liveness-detector
```

3. Run the following command in your terminal
```
$ pip install -r requirements.txt
```

4. Build the Face Detector library
```
$ cd face_det
$ sh build.sh
```

## Usage :zap:

Run the following command in your terminal

```
$ python3 main.py
```

## Contribution :zap:

Feel free to **file a new issue** with a respective title and description on the the [face-liveness-detector](https://github.com/Prem95/face-liveness-detector/issues) repository.

## Feature Request :zap:

Please also submit a pull request for any issues that might appear or any enhancements/features that could make this project perform better. **I would love to review your pull request**!

## Code of Conduct :+1:

You can find our Code of Conduct [here](/CODE_OF_CONDUCT.md).

## License :+1:
All rights reserved according to MIT © [Prem Kumar](https://github.com/Prem95/face-liveness-detector/blob/master/LICENSE)
