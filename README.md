# Hand Collages #

This project leverages the [forked repo](https://github.com/victordibia/handtracking) and [simple centroid tracking](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) to generate clips containing hands in a source video.


### Setup ###

You'll need tensorflow, the tensorflow object detection library, and OpenCV.

Install the first two by following [these instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

I installed OpenCV 4.0.0 by following [these instructions](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/), and activated the virtualenv that they instruct you to use.

Note: this can take a while

### Run ###

As with the original repo, the script to run is `detect_single_threaded.py`. You'll need to provide just one required argument, the output folder path for the video clips. Most of the time you'll also want to provide a source video file, otherwise it defaults to the webcam. So to run it on a video file:

`$ python detect_single_threaded.py [output_folder] --source [video_file]`

![image](https://drive.google.com/uc?export=view&id=1YwqehEqMvlJvsRnErkLbNTyeyuu-fp4I)

By default this will show you a debug display which plays the video and shows the tracking. I was getting about 10-12 frames with this, but performance nearly doubles when you turn it off. 

`$ python detect_single_threaded.py [output_folder] --source [video_file] -ds 0` 

Another important variable to play around with is the confidence threshold, which is in range [0-1]. 

`$ python detect_single_threaded.py [output_folder] --source [video_file] -sth 0.3`

The full list of optional arguments is available with

`$ python detect_single_threaded.py -h`

Finally, the `CentroidTracker`, which handles all the tracking, also saves a clip of the hand if more than `min_frames_to_save` frames of the object exist when it `deregisters` the hand after a `maxDisappeared` number of frames. You can set both when initializing the CentroidTracker. 

### Display ###

These clips can be optionally displayed in the browser using the included Flask webserver (`server.py`) and some conversion of the mp4 clips to webm (if you want to see it in Chrome).

Clips can be converted with 

`$ for i in <output_dir>/*.mp4; do ffmpeg -i "$i" -c:v libvpx-vp9 -crf 30 -b:v 0 -b:a 128k -c:a libopus "${i%.*}.webm"; done`

which places the .webm files in the same folder as the mp4s. Move the folder to the `static/` directory.

Run the server with `python server.py`, then in the browser go to `http://localhost:<port>/?topic=<folder_name>`

![image](https://drive.google.com/uc?export=view&id=1XVz9rPEKm_WTo6hlWXfQ4gimZ_QuKNXv)

