
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


# Medical Image Segmentation Practice

This repository provides a clean implementation for medical image segmentation.

## Clone the Repository

To begin, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/abstcol/medical_image_segmentation.git
cd medical_image_segmentation
```

## Installation

Set up the project environment with the following commands:

```bash
conda create -n medicalimg python=3.12
conda activate medicalimg
pip install -r requirements.txt
```

If you encounter any package conflicts, you may try installing the dependencies manually.

## Dataset

The default dataset path is set to `data_split`. You can download the dataset using the following link: [CodaLab - Competition](https://competitions.codalab.org/competitions/17094#participate).

**Note**: The original `.nii` files are highly correlated, which may make them difficult to work with. Therefore, the dataset has been split into `.png` files using the provided `datasplit_script.py`.

The general procedure to obtain the dataset is as follows:

1.  Download the original data (note that the test data does not include labels, so only the training and validation data are used in this case).
    
2.  Update the `traindir` in `datasplit_script.py` to reflect the location of the downloaded data.
    
3.  Execute the `datasplit_script.py` to perform the conversion.
    

## Hyperparameter Tuning

You can modify the hyperparameters by editing the `args.py` file.

## Training
(You can only start the training after preparing the data!)
To train the segmentation network, execute the following command:

```bash
python medseg.py

```

## Logs & Checkpoints

Training logs and checkpoints will be automatically saved to Weights & Biases (wandb) after each experiment. Please ensure that you are logged into your wandb account before starting the experiment.

## Resume Training

This project supports resuming from checkpoints!

To resume, set the 'resume' variable in `args.py` to the path of the checkpoint you wish to continue from.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjEzNTI5MDU0OV19
-->
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Fireflyyy04"><img src="https://avatars.githubusercontent.com/u/215456877?v=4?s=100" width="100px;" alt="Fireflyyy04"/><br /><sub><b>Fireflyyy04</b></sub></a><br /><a href="https://github.com/abstcol/medical_image_segmentation/commits?author=Fireflyyy04" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/why040506"><img src="https://avatars.githubusercontent.com/u/165774680?v=4?s=100" width="100px;" alt="why040506"/><br /><sub><b>why040506</b></sub></a><br /><a href="#ideas-why040506" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!