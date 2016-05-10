import theano
import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu0")

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, LSTM, TimeDistributed, Dense
from keras.engine import merge

from ALSTM import ALSTM, RepeatTimeDistributedVector, HierarchicalSoftmax

voc_size = 35000
voc_dim = 100
middle_dim = 200
max_out = 10


input_seq = Input(shape=(None,), dtype='int32')

embedded = Embedding(voc_size, voc_dim)(input_seq)
drop_out = Dropout(0.1)(embedded)

forward = LSTM(middle_dim, return_sequences=True, consume_less='mem')(drop_out)
backward = LSTM(middle_dim, return_sequences=True, go_backwards=True)(drop_out)

sum_res = merge([forward, backward], mode='sum')

repeat = RepeatTimeDistributedVector(max_out)(sum_res)

alstm = ALSTM(voc_dim, return_sequences=True)(repeat)

dense = TimeDistributed(Dense(voc_size))(alstm)

out = TimeDistributed(HierarchicalSoftmax(levels=3))(dense)

model = Model(input_seq,out)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


