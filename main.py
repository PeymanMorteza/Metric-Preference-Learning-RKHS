import logging
from utils.logger import setup_logger
import numpy as np
import pandas as pd
from scripts.data_script import *
from train import *
import time
from datetime import datetime
import argparse
from src.RKHS_model.RKHS import *
import os


parser = argparse.ArgumentParser(description="A script with default values.")

# Add optional arguments with defaults
parser.add_argument('--radius_1', type=float, default=7.0, 
                    help='Radius of the inner circle for synthetic data (default: 7.0)')

parser.add_argument('--radius_2', type=float, default=10.0, 
                    help='Radius of the outer circle for synthetic data (default: 10.0)')

parser.add_argument('--num_points', type=int, default=5000, 
                    help='Number of data points to generate from the concentric data distribution (default: 5000)')

parser.add_argument('--noise_std', type=float, default=0.4, 
                    help='Standard deviation of noise for the concentric data distribution (default: 1.0)')

parser.add_argument('--num_samples', type=int, default=1, 
                    help='Number of samples drawn from the distribution to form the dataset (default: 1)')

parser.add_argument('--num_epochs', type=int, default=500, 
                    help='Number of epochs for training (default: 5000)')

parser.add_argument('--data_set', type=str, default='Flatlizard', 
                    help="Dataset choice for training [lizard, chameleons, Flatlizard, pokemon, NFL, synthetic] (default: 'lizard')")

parser.add_argument('--test_size', type=float, default=0.3, 
                    help="Fraction of dataset to be used as testset")

parser.add_argument('--reg_lambda', type=float, default=0.0001, 
                    help="Regularization parameter used in associated learning problem")

parser.add_argument('--kernel', type=str, default='rbf', 
                    help="Kernel choice for mapping in RKHS")

parser.add_argument('--method', type=str, default='RKHS', 
                    help="choice of simultaneous learning [vanilla,RKHS]")

parser.add_argument('--frac', type=int, default=100, 
                    help="fraction of data")

parser.add_argument('--log_dir', type=str, default="logs_1", 
                    help="directory for logging")

parser.add_argument('--experiment', type=str, default="experiments", 
                    help="experiment name")
args = parser.parse_args()



current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# Setup logging
data_set=args.data_set
#logging.basicConfig(filename='./logs/log_{}_noker_{}.log'.format(current_time,data_set), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
setup_logger(args.experiment,args.log_dir)

kernel_map={'rbf':rbf_kernel,'euc': kernel_euc,'circ':kernel_circ}
radius_1=args.radius_1
radius_2=args.radius_2
num_points=args.num_points
noise_std=args.noise_std
num_samples=args.num_samples
test_size=args.test_size
num_epochs=args.num_epochs
reg_lam=args.reg_lambda
method=args.method
kernel=kernel_map[args.kernel]
frac=args.frac

try:
    #logging.info("Generating concentric data...")
    #df = generate_concentric_data(num_points, radius_1, radius_2, noise_std)
    
    #logging.info("Sampling pairwise data...")
    #sampled_df, _, _ = sample_pair_wise_data(df, num_samples)
    #sampled_df,_,_=generate_sample_pair_wise_data(num_points=num_points, radius_1=radius_1, radius_2=radius_2, noise_std=noise_std,num_samples=num_samples)
    if method=='vanilla':
        if data_set=="synthetic":
            logging.info("Generating concentric data...")
            sampled_df,_,_=generate_sample_pair_wise_data(num_points,radius_1,radius_2,noise_std,num_samples)
        else:
            file_path = os.path.join('.', 'data', 'processed_data', f'{data_set}.csv')
            #print("LLLLLLLLL",file_path)
            sampled_df=pd.read_csv(file_path)
    else:
        if data_set=="synthetic":
            #print("LLLLLLLLL")
            sampled_df=rkhs_synthetic_data_prepare(num_points,radius_1,radius_2,noise_std,num_samples,kernel)

        else:
            #print("LLLLLLLLL")
            sampled_df=rkhs_data_prepare(data_set,frac,kernel)
    logging.info("Splitting data into train and test sets...")
    
    
    
    X_train, X_test, y_train, y_test = split_dataframe(sampled_df, test_size, random_state=None)
    
    X_tr = np.array(X_train.values.tolist())
    Y_tr = np.array(y_train)
    X_ts = np.array(X_test.values.tolist())
    Y_ts = np.array(y_test)
    
    logging.info("Starting training...")
    A_opt, u_opt = train(X_tr, Y_tr, num_epochs, loss_fn=loss_function,reg_lam=reg_lam)
    #print(A_opt)
    #print(u_opt)
    
    logging.info("Evaluating model...")
    total_loss = evaluate_on_test_data(A_opt, u_opt, X_ts, Y_ts)
    
    logging.info("Loss on test data is %f", total_loss)

except Exception as e:
    logging.error("An error occurred: %s", str(e))
