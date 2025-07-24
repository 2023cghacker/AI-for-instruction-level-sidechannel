# Instruction level sidechannel

## Introduction
Side channel instruction level reverse
1. Clock cycle (oscillation cycle/crystal oscillator cycle/one beat): usually a beat pulse or T-cycle, which is the reciprocal of the main frequency and is the most basic unit for processing operations. Represented by P
2. State cycle: Two beats are defined as one state cycle, represented by S
3. Machine cycle: For ease of management, the execution process of an instruction is often divided into several stages, each stage completing a task.
For example, taking instructions, reading from memory, writing from memory, etc., each of these tasks is called a basic operation. The time required to complete a basic operation is called the machine cycle.
4. Instruction cycle: The total time it takes for the microcontroller to retrieve an instruction from memory and execute it. Generally composed of several machine cycles
For example, one machine cycle of AT89S52 is equal to 12 clock cycles

## Software architecture
Software Architecture Description

## Installation
#### 1. key point !!!
* The project name that has been pulled down is instrument level sidechannel, which must be renamed as src in order to run,
Because the original project was placed under src, the remaining file paths can be viewed in Configuration_ 0n.py file
#### 2. Git tutorial:
* Pull:
1) Create a new folder, right-click on Git bash here
2) Git init
3) Git remote add origin git@gitee.com : cghacker/instruction level sidechannel. git
4) Git pull origin master
* Submission:
1) Git add Configuration_ 01. py (can be a file or a folder git add. represents adding all)
2) Git commit - m "Submit Today"
3) Git push u origin master


## Instructions for use
#### static
Store raw data, which is the power consumption collected from the oscilloscope. This data is saved in the form of a mat file with multiple variables inside
This folder was not uploaded on Git because it is too large.
#### Data_Extraction
The code in this folder is responsible for extracting raw data and obtaining power consumption traces available for training
#### DataFile
This folder stores various data files, including datasets, training and testing sets, neural networks, and so on
#### Figure
This folder stores some drawn images
#### GUI
Some visualization code
#### Model_MainCode
The core code of the project, including various neural network APIs, training and testing main programs, etc
#### Program_Generation
Design assembly programs
#### Testfile
Temporary test file storage location, not very useful



## Participate in contributions
#### Author: Ling Chen
1. Fork warehouse
2. Create a new master branch
3. Submit code
4. Create a new Pull Request
