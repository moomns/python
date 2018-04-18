import pandas as pd
from pystan import StanModel
import matplotlib.pyplot as plt
import pickle

d = pd.read_csv('input/data-attendance-1.txt')
d.Score /= 200
data = d.to_dict('list')
data.update({'N':len(d)})

stanmodel = StanModel(file='model/model5-3.stan')

# NUTS (No U-Turn Sampler)
fit_nuts = stanmodel.sampling(data=data, n_jobs=1)
mcmc_sample = fit_nuts.extract()
mu_est = mcmc_sample['mu']

# ADVI (Automatic Differentiation Variational Inference)
fit_vb = stanmodel.vb(data=data)
vb_sample = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')
vb_sample = vb_sample.drop([0,1])
mu_est = vb_sample.filter(regex='mu\.\d+')

with open('output/model_and_result.pkl', 'wb') as f:
    pickle.dump(stanmodel, f)
    pickle.dump(fit_nuts, f)
