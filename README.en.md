# AI-for-instruction-level-sidechannel

## Introduction

Side-channel instruction-level reverse engineering  
1. **Clock cycle** (oscillation/crystal period/one beat): the reciprocal of the CPU frequency, denoted as *T* or *P*. It is the smallest unit for processing operations.  
2. **State cycle**: two beats form one state cycle, denoted as *S*.  
3. **Machine cycle**: for convenience, an instruction’s execution is divided into stages such as fetch, memory read, memory write, etc. Each stage is a basic operation, and the time to complete one is a machine cycle.  
4. **Instruction cycle**: the total time for the MCU to fetch and execute one instruction, composed of one or more machine cycles.  
   Example: AT89S52 has one machine cycle equal to 12 clock cycles.



## Installation

* The cloned folder is named `AI-for-instruction-level-sidechannel`; it is recommended to rename it to `src` before running, because the original project is located under `src`.  
* Other paths can be found in `Configuration_0n.py`.

* Environment: TBA

## Project Architecture

#### static  
Stores raw power traces captured by the oscilloscope as `.mat` files containing multiple variables. This folder is not uploaded to Git because of its large size.

#### Data_Extraction  
Scripts here extract the raw data to generate trainable power traces.

#### DataFile  
Contains various data files: datasets, train/test splits, neural-network checkpoints, etc.

#### figure  
Stores generated plots.

#### gui  
Visualization utilities.

#### Model_MainCode  
Core of the project: neural-network APIs, training/testing scripts, and main programs.

#### Program_Generation  
Assembly program design.

#### testfile  
Temporary test scripts; generally unused.

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip uninstall pillow
pip install pillow
