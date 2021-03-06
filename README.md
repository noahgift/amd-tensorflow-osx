# amd-tensorflow-osx
Experiments with AMD, Metal, Turicreate, Create ML, and Tensorflow on OS X

## Create ML with XCode

The Metal GPU management is very efficient in XCode

![Screen Shot 2020-10-04 at 11 23 07 PM](https://user-images.githubusercontent.com/58792/95037384-d697cb80-0698-11eb-97dc-bb5753310ae5.png)
<img width="960" alt="Screen Shot 2020-10-04 at 11 23 14 PM" src="https://user-images.githubusercontent.com/58792/95037386-d7306200-0698-11eb-9186-2d3c04667518.png">


## Multi-GPU

Experiments with AMD and Tensorflow on OS X

### Training on AMD Radeon RX 5700 XT

![Screen Shot 2020-10-04 at 7 43 18 PM](https://user-images.githubusercontent.com/58792/95029936-e8b64180-0679-11eb-89d7-954409452282.png)

Picture of Satured GPU:  

<img width="1908" alt="Screen Shot 2020-10-04 at 7 49 03 PM" src="https://user-images.githubusercontent.com/58792/95030049-ac371580-067a-11eb-8d78-eea2d1841eaf.png">

Approximate Time 110 seconds


### Training on AMD Rademon Pro W5700X

<img width="1181" alt="Screen Shot 2020-10-04 at 7 29 24 PM" src="https://user-images.githubusercontent.com/58792/95029789-afc99d00-0678-11eb-92ed-d427412f49a9.png">

Approximate time 87 Seconds.

### Training on CPU

Screenshot of no GPU utilized.  Comment out code for CPU only to test.

<img width="955" alt="Screen Shot 2020-10-04 at 7 55 49 PM" src="https://user-images.githubusercontent.com/58792/95030194-02f11f00-067c-11eb-9af0-440ad60a7ba7.png">

Screenshot of increased time to train model.  Approximately 20 Minutes on a 32 Core Mac Pro with SSD and 288GB of RAM.

![Screen Shot 2020-10-04 at 8 00 12 PM](https://user-images.githubusercontent.com/58792/95030227-4186d980-067c-11eb-9189-fcf58710ec33.png)


### Export Coreml to Tensorflow?

Use ONNX Tools:  https://github.com/onnx/onnxmltools
