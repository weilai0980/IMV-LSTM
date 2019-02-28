#!/usr/bin/python

# -- mv RNN --

# size of recurrent layers    
hidden_dim_dic = {}

hidden_dim_dic.update( {"plant": [200]} )
hidden_dim_dic.update( {"nasdaq": [820]} )
hidden_dim_dic.update( {"sml": [170]} )
hidden_dim_dic.update( {"pm25": [140]} )

# learning rate increases as network size 
lr_dic = {}

lr_dic.update( {"plant": 0.01} )
lr_dic.update( {"sml": 0.01} )
lr_dic.update( {"nasdaq": 0.05} )
lr_dic.update( {"pm25": 0.005} )

# batch size 
batch_size_dic = {}

batch_size_dic.update( {"plant": 64} )
batch_size_dic.update( {"nasdaq": 64} )
batch_size_dic.update( {"sml": 32} )
batch_size_dic.update( {"pm25": 64} )

# max_norm contraints
maxnorm_dic = {}

maxnorm_dic.update( {"plant": 5.0} )
maxnorm_dic.update( {"sml": 5.0} )
maxnorm_dic.update( {"nasdaq": 5.0} )
maxnorm_dic.update( {"pm25": 5.0} )

# attention type
attention_dic = {}

attention_dic.update( {"mv_full": "both-att"} )
attention_dic.update( {"mv_tensor": "both-att"} )

# loss type
loss_dic = {}

loss_dic.update( {"plant": "lk"} )
loss_dic.update( {"nasdaq": "lk"} )
loss_dic.update( {"sml": "mse"} )
loss_dic.update( {"pm25": "mse"} )