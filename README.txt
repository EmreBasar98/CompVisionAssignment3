The necessary imports as follows;

import glob
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam
import pathlib
from torch.autograd import Variable


The residual and non resiudal connection is handled in the forward function and when executing one approach, the relevant code snippets of other
is commented.

Pathway of Images to Train and Test are supposed to be :

	SceneDataSet
		Train, Validation and Test 
			airport_inside
			artstudio
			bakery
			bar
			bathroom
			bedroom
			bookstore
			bowling
			buffet
			casino
			church_inside
			classroom
			closet
			clothing_store
			computerroom
			
There are two different .py files for Part 1 and Part 2 respectively.

command line argument should be as follows : python <part#>.py
		e.g : python part1.py
		      python part2.py	 

Detailed explanations and code snippets included in report.