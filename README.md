# Capstone project of Data Engineering course
## Overview
The project is an **image-based fire detection system** which employs **FireNet** or **InceptionVx-OnFire** as the inference engine for ambient surveillance.

The system is supposed to run on a client-server network configuration on the Internet.
It is used to keep monitoring an outdoor site on the spot in real time
* to detect fire visually, live from the webcam footages captured with the edge camera and
* to infer on these footages so as to identify any flame in the images promptly at the terminal.

## Source
The inference models are open-source codes provided by Breckon at
https://github.com/tobybreckon/fire-detection-cnn/blob/master.
**FireNet** is taken as default.

**Fire.mp4**, **Fireball.mp4** and **Firewood.mp4** are used as test pictures. **Firewood** is taken as default.

## Set-up instruction
```bash
    $ cd fire-detection-cnn
    $ python final_project.py
```

It gives **output.png** as the result picture, where either '**Fire**' or '**Clear**' is put to indicate the inference result.