# ClaraVisio: Computational DeFogging via Image-to-Image Translation on a Free-Floating Fog Dataset

ClaraVisio (or Clara for short; Latin for "clear sight") builds on top of two previous attempts: [StereoFog](https://arxiv.org/abs/2312.02344) by Anton Pollock and [FogEye](https://github.com/Chan-man00/fogeye) by David Moody, Laura Parke, Chandler Welch, with the aim of collecting data and developing a framework for Image-to-Image translation (I2I) of foggy pictures. This project was conducted under the supervision of [Prof. Rajesh Menon](https://faculty.utah.edu/u0676529-Rajesh_Menon/hm/index.hml) at the [Laboratory for Optical Nanotechnologies](https://nanoptics.wordpress.com/) at the University of Utah during the summer of 2024 made possible by the University of Utah Summer Program for Undergraduate Research ([SPUR](https://our.utah.edu/research-scholarship-opportunities/spur/)). This work differs from previous research in using a novel free-floating fog dataset and a transformer-based model. 

---

### Table of Contents

- [Description](#description)
- [Image Capturing](#image-capturing)
- [Model Training](#model-training)
- [Datasets](#datasets)
- [How to Use](#how-to-use)
- [Getting Started](#getting-started)
- [Results](#results)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)
- [References](#references)
- [Appendix](#appendix)
- [Author Info](#author-info)

---

## Description

*Placeholder text for the project description.*

[Back to the top](#table-of-contents)

---

<details>
<summary><h2>Image Capturing</h2></summary>

*Placeholder text for the image capturing process.changed. *

Files are in raspberr_pi folder with the SOP

uses rclone to sync with google, configuration

to ssh into your raspberry pi 5: 

to have the script running at boot up use 

```bash
sudo crontab -e
```

added this code to bottom:
```
@reboot /path/to/python/script &
```
saved with CTRL+O and exit with CTRL+X

[Back to the top](#table-of-contents)

</details>

---

<details>
<summary><h2>Model Training</h2></summary>

*Placeholder text for the model training process.*

*install conda from website*

```bash
module use $HOME/MyModules
module load miniconda3/latest
```

to run jupyter notebooks you need to:

```bash
pip install notebook
```

[Back to the top](#table-of-contents)

</details>

---

## Datasets

*Placeholder text for the datasets.*

StereoFog images: [GDrive](https://drive.google.com/drive/folders/1Tzo1lDyHiiTZUwWrtjHaJ5GObJZZZMe1)
FogEye images: [MSOneDrive](https://uofutah-my.sharepoint.com/:f:/g/personal/u1259003_umail_utah_edu/EixKW5TDXE9NtsfGnCAcxcsB4uOTbCRi83Eg4y5iKnUHUQ) - Only Available for U of U students/staff, contact us for premission; needs cleaning (only download directories that contain raw files)
ClaraVisio images: 

place inside a datasets/SteroFog directory and unzip

```bash
apt-get install unzip
unzip file.zip
```

``` bash
python preprocess_stereofog_dataset.py --dataroot datasets/StereoFog/stereofog_images
```
need to run again to create a new split.

[Back to the top](#table-of-contents)

---

## How to Use

### Installation

```bash
git clone https://github.com/amirzarandi/claravisio
cd claravisio
python -m venv .venv
```
kill terminal then activate environment
```bash
pip install -r requirements.txt
```

### train

```bash
python train.py --dataroot datasets/StereoFog/stereofog_images_processed --name AL1 --model pix2pix --direction BtoA --gpu_ids 0 --n_epochs 25



python train.py --dataroot .\datasets\stereofog_images --name stereo_pix2pix --model pix2pix --direction BtoA --gpu_ids -1 --n_epochs 1  # gpu_ids -1 is for devices that are not cuda enabled.
python test.py --dataroot .\datasets\stereofog_images --direction BtoA --model pix2pix --name stereo_pix2pix --gpu_ids -1
```


### API Reference

*Placeholder text for the API reference.*

[Back to the top](#table-of-contents)

---

## Getting Started

*Placeholder text for getting started.*

[Back to the top](#table-of-contents)

---

## Results

*Placeholder text for results.*

[Back to the top](#table-of-contents)

---

## Limitations

*Placeholder text for limitations.*

[Back to the top](#table-of-contents)

---

## License

*Placeholder text for license information.*

[Back to the top](#table-of-contents)

---

## Citation

*Placeholder text for citation information.*

[Back to the top](#table-of-contents)

---

## References

*Placeholder text for references.*

[Back to the top](#table-of-contents)

---

## Appendix

*Placeholder text for the appendix.*

[Back to the top](#table-of-contents)

---

## Author Info

*Placeholder text for author info.*

[Back to the top](#table-of-contents)

---