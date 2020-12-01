# Bin Picking Simulation

## Video
[![Video(YouTube)](https://img.youtube.com/vi/v7mqrS9xTY4/0.jpg)](https://youtu.be/v7mqrS9xTY4)

## Abstract
This program helps to construct 3D scene for the bin picking based on the 3D pointcloud of an object.

The project takes a .STL mesh model as an input, then it will create a synthetic scene using [Point Cloud Library(PCL)](http://pointclouds.org/) for pointcloud processing and [Bullet Physics](http://bulletphysics.org/wordpress/) for physics simulation.

# Requirements

## Linux

To build this project you will need:

* Point Cloud Library ([PCL](https://pointclouds.org/downloads/#linux)) 
    ```
    $ sudo apt install libpcl-dev

* cmake

* Bullet3 (which is already compiled with the project)


### Compile and run
```
    mkdir build
    cd build
    cmake ..
    make
    ./binSceneMaker
```


## Windows
To build this project you will need:

* Visual Studio 2017

* [OpenNI2](https://s3.amazonaws.com/com.occipital.openni/OpenNI-Windows-x64-2.2.0.33.zip) 

* Point Cloud Library ([PCL 1.9.1](https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.9.1/PCL-1.9.1-AllInOne-msvc2017-win64.exe)) 

Install PCL as you wish, just don't install OpenNI2 at this stage


![Result](https://github.com/ktgiahieu/BinPickingSimulation/blob/main/images/PCL.png)

* cmake 3.18.5

* Bullet3 (which is already compiled with the project)
