# DeepDeform (CVPR'2020)

DeepDeform is an RGB-D video dataset containing over 390,000 RGB-D frames in 400 videos, with 5,533 optical and scene flow images and 4,479 foreground object masks. We also provide 149,228 sparse match annotations and 63,512 occlusion point annotations.


## Download Data

If you would like to download the DeepDeform data, please fill out [this google form] and, once accepted, we will send you the link to download the data.

[this google form]: https://docs.google.com/forms/d/e/1FAIpQLSeQ1hkCmmTiib-oQM9s21y3Tz9ojiI2zB8vZSqTZjT2DiRZ0g/viewform


## Online Benchmark

If you want to participate in the benchmark(s), you can submit your results at [DeepDeform Benchmark] website.

Currently we provide benchmarks for the following tasks:
- [Optical Flow]
- [Non-rigid Reconstruction]

By uploading your results on the test set to the [DeepDeform Benchmark] website the performance of you method is automatically evaluated on the hidden test labels, and compared to other already evaluated methods. You can decide if you want to make the evaluation results public or not.

If you want to evaluate on validation set, we provide code that is used for evaluation of specific benchmarks in directory `evaluation/`. To evaluate optical flow or non-rigid reconstruction, you need to adapt `FLOW_RESULTS_DIR` or `RECONSTRUCTION_RESULTS_DIR` in `config.py` to correspond to your results directory (that would be in the same format as for the online submission, described [here]).

In order to evaluate reconstruction, you need to compile additional C++ modules.

- Install necessary dependencies:
```
pip install pybind11
pip install Pillow
pip install plyfile
pip install tqdm
pip install scikit-image
```

- Inside the `evaluation/csrc` adapt `includes.py` to point to your `Eigen` include directory.

- Compile the code by executing the following in `evaluation/csrc`:
```
python setup.py install
```

[DeepDeform Benchmark]: http://kaldir.vc.in.tum.de/deepdeform_benchmark
[Optical Flow]: http://kaldir.vc.in.tum.de/deepdeform_benchmark/benchmark_optical_flow
[Non-rigid Reconstruction]: http://kaldir.vc.in.tum.de/deepdeform_benchmark/benchmark_reconstruction
[here]: http://kaldir.vc.in.tum.de/deepdeform_benchmark/documentation

## Data Organization

Data is organized into 3 subsets, `train`, `val`, and `test` directories, using `340-30-30` sequence split. In every subset each RGB-D sequence is stored in a directory `<sequence_id>`, which follows the following format:

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
- `train_selfsupervised.json` and `val_selfsupervised.json`: <br>Densely aligned optical and scene flow images using self-supervision (DynamicFusion pipeline) for a few sequences.	- `train_selfsupervised.json` and `val_skaldir
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

The data is released under [DeepDeform Terms of Use], and the code is release under a non-comercial creative commons license. 

[DeepDeform Terms of Use]: https://docs.google.com/forms/d/e/1FAIpQLSeQ1hkCmmTiib-oQM9s21y3Tz9ojiI2zB8vZSqTZjT2DiRZ0g/viewform
