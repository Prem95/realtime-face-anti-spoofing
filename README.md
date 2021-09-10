<h1 align="center">Realtime Face Anti-Spoofing Detection :robot:</h1>

<div align= "center"><img src="https://github.com/Prem95/face-liveness-detector/blob/main/misc/face.jpg" width="450" height="320"/>
  <h3>Realtime Face Anti Spoofing Detection with Face Detector to detect real and fake faces</h3>
</div>

![](https://komarev.com/ghpvc/?username=Prem95&style=flat-square&label=Views)
![Python](https://img.shields.io/badge/Python-v3.8+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/Contributions-Welcome-green.svg?style=flat)](https://github.com/Prem95/face-liveness-detector/issues)
[![Forks](https://img.shields.io/github/forks/Prem95/face-liveness-detector.svg?logo=github)](https://github.com/Prem95/face-liveness-detector/network/members)
[![Stargazers](https://img.shields.io/github/stars/Prem95/face-liveness-detector.svg?logo=github)](https://github.com/Prem95/face-liveness-detector/stargazers)

<div align= "center"><img src="https://github.com/Prem95/face-liveness-detector/blob/main/misc/demo.gif" width="600" height="550"/></div>

<h2>Please star this repo if it is useful for you!:star2: </h2>
<br/><br/>

## Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [1.1] - 10/09/2021

### Added

- Added realtime bluriness detector based on OpenCV


## [1.0] - 03/09/2021

### Added

- First commit with Face Detector, updated README
- Fixed minor issues with models not loading


## Why Build This? :thinking:
Face anti-spoofing systems has lately attracted increasing attention due to its important role in securing face recognition systems from fraudulent attacks. This project aims to provide a starting point in recognising real and fake faces based on a model that is trained with publicly available dataset


## Where to use? :hammer:
This Face Anti Spoofing detector can be used in many different systems that needs realtime facial recognition with facial landmarks. Potentially could be used in security systems, biometrics, attendence systems and etc.

Can be integrated with hardware systems for application in offices, schools, and public places for various use cases.

## Datasets and Library :green_book:

The model is trained using Tensorflow from publicly available datasets. Below listed are the data sources that the model is trained on:

CASIA: https://github.com/namtpham/casia2groundtruth

OULU: https://sites.google.com/site/oulunpudatabase/

NUAA: http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html

3DDFA: https://github.com/cleardusk/3DDFA (Face Detector Library)

Please obtain the necessary permissions before using the datasets as above.

## Prerequisites :umbrella:

All the required libraries are included in the file ```requirements.txt```. Tested on Ubuntu 20.04 with Python3.8.
Face Detector library, 3DDFA aka (```face_det```) is added as part of the repo for easy development.


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

Note: Current Face Anti Spoofing threshold is set at a value of **0.70**. This can be finetuned based on different situations as needed.

## Contribution :zap:

Feel free to **file a new issue** with a respective title and description on the the [face-liveness-detector](https://github.com/Prem95/face-liveness-detector/issues) repository.

## Feature Request :zap:

Please also submit a pull request for any issues that might appear or any enhancements/features that could make this project perform better. **I would love to review your pull request**!

## Code of Conduct :+1:

You can find our Code of Conduct [here](/CODE_OF_CONDUCT.md).

## License :+1:
All rights reserved according to MIT © [Prem Kumar](https://github.com/Prem95/face-liveness-detector/blob/master/LICENSE)
