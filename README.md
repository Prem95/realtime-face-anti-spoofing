<h1 align="center">Realtime Face Anti-Spoofing Detection</h1>

<div align= "center"><img src="https://github.com/Prem95/face-liveness-detector/blob/main/misc/face.jpg" width="350" height="250"/>
  <h4>Realtime Face Anti Spoofing Detection with Face Detector to detect real and face faces. Built using Tensorflow, Keras and OpenCV</h4>
</div>

![Python](https://img.shields.io/badge/Python-v3.8+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/Contributions-Welcome-green.svg?style=flat)](https://github.com/Prem95/face-liveness-detector/issues)
[![Forks](https://img.shields.io/github/forks/Prem95/face-liveness-detector.svg?logo=github)](https://github.com/Prem95/face-liveness-detector/network/members)
[![Stargazers](https://img.shields.io/github/stars/Prem95/face-liveness-detector.svg?logo=github)](https://github.com/Prem95/face-liveness-detector/stargazers)


![Actual Demo](https://github.com/Prem95/face-liveness-detector/blob/main/misc/demo.gif)


## Why Build This?
Face anti-spoofing systems has lately attracted increasing attention due to its important role in securing face recognition systems from fraudulent attacks. This project aims to provide a starting point in recognising real and fake faces based on a model that is trained with publicly available dataset


## Where to use?
This Face Anti Spoofing detector can possible be used in many different systems that needs realtime facial recognition with facial landmarks. Potentially could be used in security systems, biometrics, attendence systems and etc.

This integrated with hardware systems for application in offices, schools, and public places for various use cases.

## Datasets

The model is trained using Tensorflow from publicly available datasets. Below listed are the data sources that the model is trained on:

CASIA: https://github.com/namtpham/casia2groundtruth

OULU: https://sites.google.com/site/oulunpudatabase/

NUAA: http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html

Please obtain the necessary permissions before using the datasets as above.

## Prerequisites

All the required libraries are included in the file ```requirements.txt```


## Installation
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

## Usage

Run the following command in your terminal

```
$ python main.py
```

Note: Current Face Anti Spoofing threshold is set at a value of **0.70**. This can be finetuned based on different situations as needed.

## Contribution

Feel free to **file a new issue** with a respective title and description on the the [face-liveness-detector](https://github.com/Prem95/face-liveness-detector/issues) repository.

## Feature Request

Please also submit a pull request for any issues that might appear or any enhancements/features that could make this project perform better. **I would love to review your pull request**!

## Code of Conduct

You can find our Code of Conduct [here](/CODE_OF_CONDUCT.md).

## License
All rights reserved according to MIT © [Prem Kumar](https://github.com/Prem95/face-liveness-detector/blob/master/LICENSE)
