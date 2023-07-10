# 3D Scanner Simulator
Blender 3D scanner simulator for Point Cloud and RGB-D synthetic dataset creation.

## Requirements
1. Blender 2.8+ available from CLI (on Windows, `blender.exe` must be added to `PATH`)
2. Install libraries into Blender's Python via `blender -b --python install_requirements.py`

## Usage
Simply run 
```
blender -b --python simulator.py -- OUTDIR EXAMPLES -M MESHES -P PARAMS 
```

An elaborate example is 
```
blender -b --python simulator.py -- DATASET 5000 -M MODEL.obj -P PARAMS.yml -n DATASET_NAME -s 0 -t random_sphere -d 0.0 -j 0.1 -z 1.25
```
