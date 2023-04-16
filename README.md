# Tennis
Tennis Unity Environment


## Summary

- [Getting Started](#getting-started)
- [Runing the scripts](#running-the-scripts)
- [Author](#author)
- [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Install Python
I have tested this repo with Python 3.9 and 3.10. To continue, install either of these versions on your local machine. With Python installed, I suggest you create a virtual environment to install required libraries:

```bash
python -m venv desired_path_for_env
```
Activate this environment before moving to next step. For addirional help, [check Python documentation here](https://docs.python.org/3/library/venv.html).

### Install PIP Packages

The required packages for this project are listed in [requirements file](requirements.txt). To install these libraries, from the repo folder, run the following command in your virtual env:

```bash
python -m pip install -r requirements.txt
```


### Download Unity Banana Collector
The already built Unity environment for this project is accessible from following links:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here]https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


Decompress (unzip) the downloaded file and copy it to the repo folder.

## Running the scripts

The training and testing scripts are located in [scripts](scripts) folder.

### Training

To train the model, use [train_agent.py](scripts/train_agent.py) script. This script accepts the following arguments:

- Path to downloaded Unity App: --unity-app
- Target Score to save trained model: --target-score

```bash
cd scripts
python train_agent.py --unity-app Tennis.app --target-score 0.5
```


### Testing

To compare a trained agent with a untrained one, use [test_agent.py] script. This script accepts the following arguments: 

- Path to downloaded Unity App: --unity-app


## Author
  - **Sina Fathi-Kazerooni** - 
    [Website](https://sinafathi.com)


## License

This project is open source under MIT License and free to use. It is for educational purposes only and provided as is.

I have used parts of **DDPG_Pendulum** scripts in [Udacity DRL](https://github.com/udacity/deep-reinforcement-learning/) repo under [MIT License](https://github.com/udacity/deep-reinforcement-learning/blob/master/LICENSE). Scripts in [mlagents](mlagents) are based on [Udacity DRL](https://github.com/udacity/deep-reinforcement-learning/) repo with minor modifications.
