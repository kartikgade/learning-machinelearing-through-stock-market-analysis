import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

# Load data
data = pd.read_csv('HDFC.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preprocess data
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Define features and targets
features = data[['Open', 'High', 'Low', 'Volume']].values
targets = data['Returns'].values

# Define Bayesian linear regression model
with pm.Model() as model:
    # Define priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=len(features))
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Define likelihood
    y_obs = pm.Normal('y_obs', mu=alpha + pm.math.dot(beta, features.T), sigma=sigma, observed=targets)

    # Run inference
    trace = pm.sample(1000, tune=1000, chains=4)

# Define new features for prediction
new_features = np.array([[3000, 3050, 2950, 1000000], [3050, 3100, 3000, 1500000]])

# Define posterior predictive distribution
with model:
    y_pred = pm.Normal('y_pred', mu=alpha + pm.math.dot(beta, new_features.T), sigma=sigma)

    # Draw samples from posterior predictive distribution
    y_samples = pm.sample_posterior_predictive(trace, samples=1000)['y_pred']

# Plot posterior predictive distribution
sns.kdeplot(y_samples.flatten(), label='Posterior Predictive')
plt.xlabel('Target')
plt.ylabel('Density')
plt.title('Posterior Predictive Distribution')
plt.show()
