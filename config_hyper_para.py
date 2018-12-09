#!/usr/bin/python

# -- mv --

# size of recurrent layers    
hidden_dim_dic = {}

hidden_dim_dic.update( {"plant": [200]} )
hidden_dim_dic.update( {"plant_pearson": [120]} )
hidden_dim_dic.update( {"plant_no_target": [180]} )
hidden_dim_dic.update( {"plant_sub_full": [120]} )
hidden_dim_dic.update( {"plant_sub_tensor": [120]} )
hidden_dim_dic.update( {"plant_uni": [128]} )

hidden_dim_dic.update( {"nasdaq": [820]} )
hidden_dim_dic.update( {"nasdaq_pearson": [410]} )
hidden_dim_dic.update( {"nasdaq_no_target": [810]} )
hidden_dim_dic.update( {"nasdaq_sub_full": [410]} )
hidden_dim_dic.update( {"nasdaq_sub_tensor": [410]} )
hidden_dim_dic.update( {"nasdaq_uni": [128]} )

# lk: ? mse: 170
hidden_dim_dic.update( {"sml": [170]} )
hidden_dim_dic.update( {"sml_pearson": [90]} )
hidden_dim_dic.update( {"sml_no_target": [160]} )
hidden_dim_dic.update( {"sml_sub_full": [90]} )
hidden_dim_dic.update( {"sml_sub_tensor": [90]} )
hidden_dim_dic.update( {"sml_uni": [128]} )


# learning rate increases as network size 
lr_dic = {}

# lk: 0.01, mse: 0.005
lr_dic.update( {"plant": 0.01} )
lr_dic.update( {"plant_pearson": 0.005} )
lr_dic.update( {"plant_no_target": 0.005} )
lr_dic.update( {"plant_sub_full": 0.005} )
lr_dic.update( {"plant_sub_tensor": 0.005} )
lr_dic.update( {"plant_uni": 0.005} )

# lk: ?, mse: 0.01
lr_dic.update( {"sml": 0.01} )
lr_dic.update( {"sml_pearson": 0.01} )
lr_dic.update( {"sml_no_target": 0.01} )
lr_dic.update( {"sml_sub_full": 0.01} )
lr_dic.update( {"sml_sub_tensor": 0.01} )
lr_dic.update( {"sml_uni": 0.01} )

lr_dic.update( {"nasdaq": 0.05} )
lr_dic.update( {"nasdaq_pearson": 0.05} )
lr_dic.update( {"nasdaq_no_target": 0.05} )
lr_dic.update( {"nasdaq_sub_full": 0.05} )
lr_dic.update( {"nasdaq_sub_tensor": 0.05} )
lr_dic.update( {"nasdaq_uni": 0.05} )

'''
lr_dic.update( {"pm25": 0.002} )
lr_dic.update( {"pm25": 0.002} )
lr_dic.update( {"pm25_sub_full": 0.002} )
lr_dic.update( {"pm25_sub_tensor": 0.002} )
'''

# batch size 
batch_size_dic = {}
batch_size_dic.update( {"plant": 64} )
batch_size_dic.update( {"plant_pearson": 64} )
batch_size_dic.update( {"plant_no_target": 64} )
batch_size_dic.update( {"plant_sub_full": 64} )
batch_size_dic.update( {"plant_sub_tensor": 64} )
batch_size_dic.update( {"plant_uni": 64} )

batch_size_dic.update( {"nasdaq": 64} )
batch_size_dic.update( {"nasdaq_pearson": 64} )
batch_size_dic.update( {"nasdaq_no_target": 64} )
batch_size_dic.update( {"nasdaq_sub_full": 64} )
batch_size_dic.update( {"nasdaq_sub_tensor": 64} )
batch_size_dic.update( {"nasdaq_uni": 64} )

#  lk ?, mse 32
batch_size_dic.update( {"sml": 32} )
batch_size_dic.update( {"sml_pearson": 32} )
batch_size_dic.update( {"sml_no_target": 32} )
batch_size_dic.update( {"sml_sub_full": 32} )
batch_size_dic.update( {"sml_sub_tensor": 32} )
batch_size_dic.update( {"sml_uni": 32} )

'''
batch_size_dic.update( {"pm25": 32} )
batch_size_dic.update( {"pm25": 32} )
batch_size_dic.update( {"pm25_no_target": 32} )
batch_size_dic.update( {"pm25_sub_full": 32} )
batch_size_dic.update( {"pm25_sub_tensor": 32} )
'''

# max_norm contraints
maxnorm_dic = {}

maxnorm_dic.update( {"plant": 5.0} )
maxnorm_dic.update( {"plant_pearson": 5.0} )
maxnorm_dic.update( {"plant_no_target": 5.0} )
maxnorm_dic.update( {"plant_sub_full": 5.0} )
maxnorm_dic.update( {"plant_sub_tensor": 5.0} )
maxnorm_dic.update( {"plant_uni": 5.0} )

maxnorm_dic.update( {"sml": 5.0} )
maxnorm_dic.update( {"sml_pearson": 5.0} )
maxnorm_dic.update( {"sml_no_target": 5.0} )
maxnorm_dic.update( {"sml_sub_full": 5.0} )
maxnorm_dic.update( {"sml_sub_tensor": 5.0} )
maxnorm_dic.update( {"sml_uni": 5.0} )

maxnorm_dic.update( {"nasdaq": 5.0} )
maxnorm_dic.update( {"nasdaq_pearson": 5.0} )
maxnorm_dic.update( {"nasdaq_no_target": 5.0} )
maxnorm_dic.update( {"nasdaq_sub_full": 5.0} )
maxnorm_dic.update( {"nasdaq_sub_tensor": 5.0} )
maxnorm_dic.update( {"nasdaq_uni": 5.0} )

'''
maxnorm_dic.update( {"pm25": 4.0} )
maxnorm_dic.update( {"pm25_sub_full": 4.0} )
maxnorm_dic.update( {"pm25_sub_tensor": 4.0} )
'''

# attention type
attention_dic = {}
attention_dic.update( {"plain": "temp"} )
attention_dic.update( {"mv_full": "both-att"} )
attention_dic.update( {"mv_tensor": "both-att"} )

'''
attention_dic.update( {"clstm": ""} )
attention_dic.update( {"clstm_sub": ""} )

attention_dic.update( {"sep": "both-att"} )
attention_dic.update( {"sep_sub": "both-att"} )
'''

# loss type
loss_dic = {}
loss_dic.update( {"plant": "lk"} )
loss_dic.update( {"plant_pearson": "lk"} )
loss_dic.update( {"plant_no_target": "lk"} )
loss_dic.update( {"plant_sub_full": "lk"} )
loss_dic.update( {"plant_sub_tensor": "lk"} )
loss_dic.update( {"plant_uni": "lk"} )

loss_dic.update( {"nasdaq": "lk"} )
loss_dic.update( {"nasdaq_pearson": "lk"} )
loss_dic.update( {"nasdaq_no_target": "lk"} )
loss_dic.update( {"nasdaq_sub_full": "lk"} )
loss_dic.update( {"nasdaq_sub_tensor": "lk"} )
loss_dic.update( {"nasdaq_uni": "lk"} )

loss_dic.update( {"sml": "mse"} )
loss_dic.update( {"sml_pearson": "mse"} )
loss_dic.update( {"sml_no_target": "mse"} )
loss_dic.update( {"sml_sub_full": "mse"} )
loss_dic.update( {"sml_sub_tensor": "mse"} )
loss_dic.update( {"sml_uni": "mse"} )