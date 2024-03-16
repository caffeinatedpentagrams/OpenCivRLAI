# OpenCivRLAI

## Build the docker image
```sh
$ git clone https://github.com/caffeinatedpentagrams/OpenCivRLAI.git
$ cd OpenCivRLAI
$ docker build -t freeciv-gui-mp .
```

## RL agent Python virtual environment setup
```sh
$ cd PythonClient
$ python -m venv venv # create the virtual environment
$ source venv/bin/activate # activate the virtual environment
$ pip install -r requirements.txt # install dependencies
```

## Run RL agent
Make sure the Python virtual environment is activated.
```sh
$ cd PythonClient
$ python main.py
```

## Obtain freeciv-server v3.0.10
Make sure you have a working executable of `freeciv-server`. One that runs on mac is provided in the repo, and for windows users there are easy ready to download links. See the wiki page for detailed instructions on running the complete set of code/processes.
