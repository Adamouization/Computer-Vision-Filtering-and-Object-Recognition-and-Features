# Computer Vision: Filtering, Object Recognition & Features

**Intensity-based** template matching and **feature-based** template matching using **SIFT** algorithms for matching images are implemented. A [Training dataset](https://github.com/Adamouization/Computer-Vision-Coursework/tree/master/dataset/Training) of images (icons) a [Testing dataset](https://github.com/Adamouization/Computer-Vision-Coursework/tree/master/dataset/Test) (various combinations of icons) as shown in Figure 2 are used.

Project developed in collaboration with [Andreak Lissak](https://github.com/yissok).

The report can be read [here](https://github.com/Adamouization/Computer-Vision-Coursework/blob/master/report/report.pdf).

## Usage

Clone the repository (or download the zipped project):
`$ git clone https://github.com/Adamouization/Computer-Vision-Coursework`

Create a virtual environment for the project and activate it:

```
virtualenv ~/Environments/Computer-Vision-Coursework
source Computer-Vision-Coursework/bin/activate
```

Once you have the virtualenv activated and set up, `cd` into the project directory and install the requirements needed to run the app:

```
pip install -r requirements.txt
```

You can now run the app:
```
python main.py -m <model_type> --mode <mode> --debug
```

where:
* `-m <model_type>` corresponds to the matching technique to use e.g. `convolution`, `intensity` or `sift`.
* `--mode <mode>` corresponds to `train` or `test`.
* `--d` runs the program in debug mode with additional print statements.

## Contact
* Emails: adam@jaamour.com & andrea.lissak@gmail.com
* LinkedIn: [@adamjaamour](https://www.linkedin.com/in/adamjaamour/) & [@andreaklissak](https://www.linkedin.com/in/andrea-lissak-3bbb88129/) 
