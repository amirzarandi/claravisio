# ClaraVisio: Computational DeFogging via Image-to-Image Translation on a Novel Free-Floating Fog Dataset

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

ClaraVisio (Latin for "clear vision") is an innovative Image-to-Image (I2I) translation framework designed to transform foggy images into clear ones, addressing critical challenges in autonomous vehicle technology and computer vision applications in adverse weather conditions.

## Key Features

- **Novel Free-Floating Fog Dataset**: First-of-its-kind dataset with 1,144 paired images of free-floating fog, more closely mimicking real-world conditions
- **Custom Data Collection Setup**: Raspberry Pi-based system with 3D-printed funnel for consistent fog capture
- **Advanced I2I Translation**: Pix2Pix conditional GAN model optimized for fog removal
- **Comprehensive Evaluation**: Multiple metrics including MS-SSIM, CW-SSIM, PSNR, and custom fog density measures
- **High Performance**: Achieved 91% MS-SSIM score on defogging tasks

## Research Background

Traditional fog removal datasets rely on entrapped fog, which fails to capture the complexity of real-world fog behavior. ClaraVisio introduces the first dataset using free-floating fog, collected at the University of Utah campus in July 2024. This approach provides more realistic training data for autonomous vehicle applications and computer vision systems operating in adverse weather.

**Key Innovations:**
- Free-floating fog vs. traditional entrapped fog approaches
- Custom hardware setup for consistent paired image collection
- Optimized pix2pix model with MS-SSIM loss function
- Comprehensive fog density analysis using variance of Laplacian

## Dataset Characteristics

- **Total Images**: 1,144 paired clear/foggy images
- **Collection Period**: July 2024
- **Location**: University of Utah campus, Salt Lake City
- **Fog Density Range**: Quantified using variance of Laplacian (vL)
- **Image Resolution**: 320x240 (processed from higher resolution captures)
- **Dataset Split**: 80% training, 10% validation, 10% testing

### Fog Density Distribution
- **Peak Distribution**: Around 100 vL (92.13% of data points after outlier removal)
- **Non-homogeneity Score**: 0.877 (vs. 0.493 for StereoFog)
- **Edge Density**: 0.0223 (10x higher than comparable datasets)

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Environment Setup

```bash
git clone https://github.com/amirzarandi/claravisio
cd claravisio
```

Create a virtual environment:
```bash
# Using conda
conda create --name claravisio python=3.9.7
conda activate claravisio

# Or using pyenv
pyenv virtualenv 3.9.7 claravisio
pyenv activate claravisio
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `torch` - PyTorch deep learning framework
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `PIL` - Image processing
- `Augmentor` - Data augmentation
- `ssim` - Structural similarity metrics
- `pytorch-msssim` - Multi-scale SSIM

## Dataset Setup

### Download Datasets
- **ClaraVisio Dataset**: Contact authors for access
- **StereoFog Dataset**: [Google Drive](https://drive.google.com/drive/folders/1Tzo1lDyHiiTZUwWrtjHaJ5GObJZZZMe1)
- **FogEye Dataset**: [MS OneDrive](https://uofutah-my.sharepoint.com/:f:/g/personal/u1259003_umail_utah_edu/EixKW5TDXE9NtsfGnCAcxcsB4uOTbCRi83Eg4y5iKnUHUQ) (U of U access only)

### Data Preprocessing

```bash
# For StereoFog dataset
python preprocess_stereofog_dataset.py --dataroot /path/to/stereofog_images --augment

# For ClaraVisio dataset
python preprocess_clara.py --dataroot /path/to/claravisio_images

# Convert PNG to BMP format
python png2bmp.py --dataroot /path/to/images --output_name processed_images

# Create aligned dataset for pix2pix
python make_dataset_aligned.py --dataset-path /path/to/processed/dataset
```

## Usage

### Training

Train a pix2pix model on ClaraVisio dataset:
```bash
python train.py \
    --dataroot /path/to/claravisio_processed \
    --name clara_model \
    --model pix2pix \
    --direction BtoA \
    --gpu_ids 0 \
    --n_epochs 35 \
    --n_epochs_decay 25 \
    --norm batch \
    --netG resnet_9blocks \
    --netD n_layers \
    --n_layers_D 2 \
    --gan_mode vanilla \
    --ngf 128 \
    --ndf 32 \
    --lr_policy linear \
    --init_type normal
```

### Testing

Evaluate trained model:
```bash
python test.py \
    --dataroot /path/to/test/dataset \
    --name clara_model \
    --model pix2pix \
    --direction BtoA \
    --gpu_ids 0
```

### Evaluation

Generate quantitative metrics:
```bash
python quantitative_evaluation_model_results.py --results_path results/clara_model
```

Plot visual results:
```bash
python plot_model_results.py \
    --results_path results/clara_model \
    --num_images 5 \
    --shuffle
```

Monitor training progress:
```bash
python plot_epoch_progress.py \
    --checkpoints_path checkpoints/clara_model \
    --model_name clara_model
```

## Hardware Setup (Data Collection)

### Raspberry Pi Configuration

Components needed:
- Raspberry Pi 5
- Compatible camera module
- GPIO-connected shutter button
- Status LEDs
- Power bank
- 3D-printed funnel (CAD files included)

Setup instructions:
1. Install required Python libraries: `gpiod`, `picamera2`, `gpiozero`
2. Configure `rclone` for Google Drive sync
3. Set up auto-start with crontab:
   ```bash
   sudo crontab -e
   # Add: @reboot /path/to/python/script &
   ```

The custom collection script (`main_pilot.py`) handles:
- Automated image capture workflow
- Real-time cloud synchronization  
- LED status indicators
- Logging and error handling

## Results

### Model Performance
- **MS-SSIM**: 0.91 (91%)
- **CW-SSIM**: 0.86 (86%)
- **SSIM**: 0.78 (78%)
- **Pearson Correlation**: 0.36
- **NCC**: 0.95 (95%)

### Dataset Comparison
ClaraVisio vs. StereoFog:
- **10x higher edge density** preserving structural information
- **More varied blur patterns** across images (0.877 vs 0.493 non-homogeneity)
- **Higher overall blur and variation** representing realistic fog conditions

## Applications

- **Autonomous Vehicles**: Enhanced perception in foggy conditions
- **Traffic Safety Systems**: Improved visibility for monitoring systems  
- **Computer Vision**: Robust performance in adverse weather
- **Security Systems**: Clear imaging in low-visibility scenarios

## Citation

If you use ClaraVisio in your research, please cite:

```bibtex
@article{zarandi2024claravisio,
  title={ClaraVisio: Computational DeFogging via Image-to-Image Translation on a Novel Free-Floating Fog Dataset},
  author={Zarandi, Amir and Menon, Rajesh},
  journal={University of Utah Summer Program for Undergraduate Research (SPUR)},
  year={2024}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Prof. Rajesh Menon** - Faculty Mentor, Laboratory for Optical Nanotechnologies
- **University of Utah SPUR Program** - Summer Program for Undergraduate Research
- **Previous Work**: StereoFog by Anton Pollock, FogEye by Moody, Parke, and Welch
- **PyTorch CycleGAN/Pix2Pix Framework** - Base implementation

## Contact

- **Author**: Amir Zarandi
- **Institution**: Columbia University

## Future Work

- Expand dataset with diverse locations beyond university campus
- Explore alternative I2I architectures (CycleGAN, AttentionGAN)
- Test model performance on real-world fog scenarios
- Develop failure detection methods
- Improve robustness for object detection and text recognition
- Integration with autonomous vehicle perception pipelines

---

*ClaraVisio represents a significant step forward in computational defogging, providing the research community with novel datasets and methodologies for addressing visibility challenges in autonomous systems.*
