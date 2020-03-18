# DeepDeform (CVPR'2020)

DeepDeform is an RGB-D video dataset containing over 390k frames in 400 videos, with [INSERT FINAL] optical and scene flow images and [INSERT FINAL] foreground object masks. Furthermore, we also provide [INSERT FINAL] sparse match annotations and [INSERT FINAL] occlusion point annotations.


## Download Data

If you would like to download the DeepDeform data, please fill out [this google form] and, once accepted, we will send you the link to download the data.

[this google form]: https://docs.google.com/forms/d/e/1FAIpQLSeQ1hkCmmTiib-oQM9s21y3Tz9ojiI2zB8vZSqTZjT2DiRZ0g/viewform


## Data Organization

Data is organized into 3 subsets, `train`, `val` and `test` directories, using `340-30-30` sequence split. In every subset each RGB-D sequence is stored in a directory `<sequence_id>`, which follows the following format:

```
<sequence_id>
|-- <color>: color images for every frame (`%06d.jpg`)
|-- <depth>: depth images for every frame (`%06d.png`)
|-- <mask>: mask images for a few frames (`%06d.png`)
|-- <optical_flow>: optical flow images for a few frame pairs (`<object_id>_<source_id>_<target_id>.oflow` or `%s_%06d_%06d.oflow`)
|-- <scene_flow>: scene flow images for a few frame pairs (`<object_id>_<source_id>_<target_id>.sflow` or `%s_%06d_%06d.sflow`)
|-- <intrinsics.txt>: 4x4 intrinsics matrix
```

All labels are provided in `.json` files in root dataset r directory:
- `train_matches.json` and `val_matches.json`: <br>Manually annotated sparse matches.
- `train_dense.json` and `val_dense.json`: <br>Densely aligned optical and scene flow images with the use of sparse matches as a guidance.
- `train_selfsupervised.json` and `val_selfsupervised.json`: <br>Densely aligned optical and scene flow images using self-supervision (DynamicFusion pipeline) for a few sequences.
- `train_masks.json` and `val_masks.json`: <br>Dynamic object annotations for a few frames per sequence.
- `train_occlusions.json` and `val_occlusions.json`: <br>Manually annotated sparse occlusions.


## Data Formats

We recommend you to test out scripts in `demo/` directory in order to check out loading of different file types.

**RGB-D Data**: 3D data is provided as RGB-D video sequences, where color and depth images are already aligned. Color images are provided as 8-bit RGB .jpg, and depth images as 16-bit .png (divide by 1000 to obtain depth in meters).

**Camera Parameters**: A 4x4 intrinsic matrix is given for every sequence (because different cameras were used for data capture, every sequence can have different intrinsic matrix). Since the color and depth images are aligned, no extrinsic transformation is necessary.

**Optical Flow Data**: Dense optical flow data is provided as custom binary image of resolution 640x480 with extension .oflow. Every pixel contains two values for flow in x and y direction, in pixels. Helper function to load/store binary flow images is provided in `utils.py`.

**Scene Flow Data**: Dense scene flow data is provided as custom binary image of resolution 640x480 with extension .sflow. Every pixel contains 3 values for flow in x, y and z direction, in meters. Helper function to load/store binary flow images is provided in `utils.py`.

**Object Mask Data**: A few frames per sequences also include foreground dynamic object annotation. The mask image is given as 16-bit .png image (1 for object, 0 for background).

**Sparse Match Annotations**: We provide manual sparse match annotations for a few frame pairs for every sequence. They are stored in .json format, with paths to corresponding source and target RGB-D frames, as a list of source and target pixels.

**Sparse Occlusion Annotations**: We provide manual sparse occlusion annotations for a few frame pairs for every sequence. They are stored in .json format, with paths to corresponding source and target RGB-D frames, as a list of occluded pixels in source frame.


## Benchmark Tasks

In directory `evaluation/` we provide code that is used for evaluation of specific benchmarks:
- Optical Flow
- Non-rigid Reconstruction

If you want to participate in the benchmark(s), you can submit your results at [DeepDeform Benchmark] website.

[DeepDeform Benchmark]: http://mars.vc.in.tum.de/deepdeform_benchmark

To evaluate flow or reconstruction, you need to adapt `FLOW_RESULTS_DIR` or `RECONSTRUCTION_RESULTS_DIR` in `config.py` to correspond to your results directory (that would be in the same format as published online for submission).

In order to evaluate reconstruction, you need to compile additional C++ modules.

- Inside the `evaluation/csrc` adapt `includes.py` to point to your `Eigen` include directory.
- Compile the code by executing the following in `evaluation/csrc`:
```
python setup.py install
```


## Citation

If you use DeepDeform data or code please cite:
```
@inproceedings{bozic2020deepdeform, 
    title={DeepDeform: Learning Non-rigid RGB-D Reconstruction with Semi-supervised Data}, 
    author={Bo{\v{z}}i{\v{c}}, Alja{\v{z}} and Zollh{\"o}fer, Michael and Theobalt, Christian and Nie{\ss}ner, Matthias}, 
    journal={Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    year={2020}
}
```


## Help

If you have any questions, please contact us at deepdeform@googlegroups.com, or open an issue at Github.


## License

The data is released under [DeepDeform Terms of Use], and the code is release under the MIT license. 

[DeepDeform Terms of Use]: https://docs.google.com/forms/d/e/1FAIpQLSeQ1hkCmmTiib-oQM9s21y3Tz9ojiI2zB8vZSqTZjT2DiRZ0g/viewform
