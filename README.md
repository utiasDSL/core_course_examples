# Optimal Control and Decision-Making 
This repository contains the examples shown in the course 'Optimal Control and Decision-Making' by Prof. Angela P. Schoellig at TUM. You can try them yourself following the steps below. The examples are implemented in Python and use Jupyter notebooks for in-depth exploration of the concepts covered in the lectures. 

## Course Overview
These examples complement the theoretical concepts covered in the lectures by providing interactive implementations. You'll find implementations of dynamic programming, model predictive control, reinforcement learning, and more - applied to the [mountain car problem](https://en.wikipedia.org/wiki/Mountain_car_problem). 

![Figure 1: The mountain car reaches the top of the hill using nonlinear model predictive control (MPC)](../figure/MountainCarAnimation.gif)

## Prerequisites
- Basic knowledge of Python programming
- Familiarity with numerical methods and linear algebra
- Understanding of control theory fundamentals
- Previous experience with Jupyter notebooks is helpful but not required

## Repository Structure
```
├── Dockerfile
├── ex1_DP/                   # DP examples
│   ├── dynamic_programming.ipynb
│   └── ...
├── ex2_LQR/                  # LQR examples
│   ├── linear_quadratic_regulator.ipynb
│   └── ...
└── ... # More examples
```

## Setup

### Docker Desktop
These instructions use Docker Desktop to create a containerized environment using Docker that works on any common operating systems (OS), e.g., Linux, Windows, and Mac OS. If you are on Linux you may instead directly install the Docker Engine, however, we recommend using Docker Desktop. 

Install Docker Desktop for your OS: 
https://docs.docker.com/desktop/ 

### Visual Studio Code
These instructions are for Visual Studio (VS) Code and have only been tested with VS Code. 

Download and install VS Code for your OS:
https://code.visualstudio.com/Download

VS Code lets us conveniently open a directory inside a Docker container. For this, install the Dev Containers extension in VS Code. See the instructions here: https://code.visualstudio.com/docs/devcontainers/tutorial 

### Git & GitHub
For version control we are using Git. Install Git on your system (if you don't have it already): https://git-scm.com/downloads  

We host this repository on GitHub. Make sure that you have an account to pull it (and file issues, create pull requests, etc.): https://docs.github.com/en/get-started  

You will have to set up your account to connect to GitHub via SSH: https://docs.github.com/en/authentication/connecting-to-github-with-ssh. In particular, you will have to add a SSH key to your GitHub account (if you haven't done so already). 

If you are unfamiliar with Git, check out this brief overview: https://education.github.com/git-cheat-sheet-education.pdf 

## Installation 
First, pull the repository
```
cd <YOUR DESIRED DIRECTORY>
git clone git@github.com:utiasDSL/core_course_examples.git
```
Open VS Code. Press the `F1` key, select `Dev Containers: Open Folder in Container` in the search bar, and select the cloned repository as the folder. The repository contains a `Dockerfile`, which is like a recipe how to set up all the required dependencies in a container. The creation of the container may take a couple of minutes.  

## Usage
In VS Code's Explorer, expand the folder `ex1_DP` and double-click on the Jupyter notebook `1.1_dynamic_programming.ipynb`.

This will open up the first example on dynamic programming and will prompt you to install additional extensions, e.g., the Jupyter and Python extensions (if you haven't installed them already). To run the cells in the notebook, you will have to select a Kernel. As the kernel, select the Python environment provided by the Docker container. Detailed instructions on how to use Jupyter notebooks in VS Code can be found here: https://code.visualstudio.com/docs/datascience/jupyter-notebooks 

Over time we will push updates to this repository, e.g., fixes and additional examples. You can receive them by using
``` 
git pull 
```

If you run into any issues, feel free to create an issue on GitHub. We recommend creating your own fork of the repository if you are planning to do development on your own, see: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks 

## Authors
Haocheng Zhao, Lukas Brunke, SiQi Zhou, and Angela P. Schoellig from the Learning Systems and Robotics Lab @ TUM