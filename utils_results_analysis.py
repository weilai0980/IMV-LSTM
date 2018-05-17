#!/usr/bin/python

from utils_libs import *

# MV-LSTM 


# --- prior 

def prior(df, num_vari, label_vari):
    
    print '\n --- individual prior mean and variance:'
    indi_prior = []
    mean_prior = []
    
    for z in range(num_vari):
        tmp_prior = df[str(z)]
        indi_prior.append(tmp_prior)
        mean_prior.append( mean(tmp_prior) ) 
        
    vari_prior = zip( label_vari, mean_prior )
    print sorted(vari_prior, key = lambda x:x[1]) 
    
    
    return np.transpose(indi_prior)
        
# --- aggregated prior

def aggregate_prior(df, num_vari, label_vari):
    
    logit = []
    for z in range(num_vari):
        logit.append(sum(list(df[str(z)])))
    
    tmpsum = sum(logit)
    agg_prior = [i/(tmpsum+1e-5) for i in logit]
    
    print '\n --- aggregated prior :'
    
    vari_prior = zip( label_vari, agg_prior )
    print sorted(vari_prior, key = lambda x:x[1])
    
#     for z in range(num_vari):
#         print label_vari[z], agg_prior[z]
        
    return agg_prior

def aggregate_poster(poster, num_vari, label_vari):
    
    logit = []
    for z in range(num_vari):
        logit.append( sum([k[z] for k in poster]) )
    
    tmpsum = sum(logit)
    agg_poster = [i/(tmpsum+1e-5) for i in logit]
    
    print '\n --- aggregated posterior :'
    
    vari_poster = zip( label_vari, agg_poster )
    print sorted(vari_poster, key = lambda x:x[1])
    
#     for z in range(num_vari):
#         print label_vari[z], agg_prior[z]
        
    return agg_poster

# --- individual posterior 

def individual_posterior(df, num_vari, label_vari):
    
    truth = list(df['truth'])
    poster = []

    for k in range(df.shape[0]):
        
        tmp_logit = []
    
        for z in range(num_vari):
            tmp_logit.append(exp(-1.0*((truth[k]-df['pre' + str(z)].iloc[k])/truth[k])**2/2.0)*df[str(z)][k])
    
        tmpsum = sum(tmp_logit)
        tmp_post = [i/(tmpsum+1e-25) for i in tmp_logit]
        
        poster.append(tmp_post)
    
    print '\n --- individual_posterior mean and variance :'
    
    mean_poster = []
    for z in range(num_vari):
        
        tmp_posterior = [i[z] for i in poster]
#         print label_vari[z], ' : ', mean(tmp_posterior), std(tmp_posterior)
        mean_poster.append(mean(tmp_posterior))

    
    vari_prior = zip( label_vari, mean_poster )
    print sorted(vari_prior, key = lambda x:x[1]) 
        
    return poster
        
# --- posterior 

def posterior(df, num_vari, label_vari, agg_prior):
    
    total_idx = range(df.shape[0])
    np.random.shuffle(total_idx)

    sample_num = 10
    sample_index = total_idx[ :sample_num ]
    sample_truth = list(df['truth'][sample_index])
    
    logit = []
    for z in range(num_vari):
        
        tmplogit = 1.0
        tmp_prior = list(df[str(z)][sample_index])
        tmp_pred  = list(df['pre' + str(z)][sample_index])
    
#         print '\n --- variable ', z
    
        for k in range(sample_num):
            
#             print sample_truth[k], tmp_pred[k], tmp_prior[k]
            tmpllk =exp(-1.0*((sample_truth[k]-tmp_pred[k])/sample_truth[k])**2/2.0)
            tmplogit *= (tmpllk )
        
    
        logit.append(tmplogit*agg_prior[z])

    print '\n--- posterior: ', logit
    sum_logit = sum(logit)
    posterior = [i*1.0/(sum_logit+1e-350) for i in logit]
    print posterior
