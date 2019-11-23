Experienced problems with weights becoming NaN at the beginning of training, solved by adding BatchNormalization after residual connections

Problems occured again on GPU, tried removing BatchNormalization and adding K.epsilon to K.sqrt in loss function