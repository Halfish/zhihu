# zhihu
知乎看山杯比赛

### Dependencies

- [zutil](https://github.com/Halfish/zutil), my util code for DL/ML
- [pytorch](http://pytorch.org): A deep learning framework with GPU & auto gradient.

### Files

- configure.py
	- load word2vec & dataset provided from competitioni, build dictionary
- data.py
	- dataset loader for neural network models, split batch
- model.py
	- baseline/LSTM/CNN models
- train.ipynb
	- training and save the model with multi-label loss
- parameter.json
	- paramters for training, json file
- inference and judge.ipynb
	- test/validation, generate and submit the final result

