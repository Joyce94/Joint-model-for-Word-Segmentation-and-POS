## Introduction
	this is an implementation for the paper:"A Simple and Effective Neural Model for Joint Word Segmentation and POS Tagging".

## Requirement
	* python 3
	* pytorch > 0.1(I used pytorch 0.3.0)
	* numpy

## Performance  Result
-----------------------

|Fscore|environment|seg|pos|
| ------|------|------- |------- |
|paper|C++|96.36|92.51|
|my_version_nobatch|pytorch|96.46|92.41|
|my_version_batch|pytorch|96.34|92.21|

-----------------------

## Data
	CTB6.0
