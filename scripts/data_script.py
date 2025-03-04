import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.RKHS_model.RKHS import *
import os 


# def generate_concentric_data(num_points=5000, radius_1=5, radius_2=10, noise_std=0.4):
#     """
#     Generates a dataset of points arranged in two concentric circles with Gaussian noise.
    
#     Parameters:
#     - num_points (int): Total number of data points to generate. Each circle segment will have num_points / 2 points.
#     - radius_1 (float): Radius of the smaller circle.
#     - radius_2 (float): Radius of the larger circle.
#     - noise_std (float): Standard deviation of the Gaussian noise added to the data points.
    
#     Returns:
#     - pd.DataFrame: A DataFrame containing the generated data with columns 'X', 'Y', and 'Label'.
#       - 'X': X-coordinates of the data points.
#       - 'Y': Y-coordinates of the data points.
#       - 'Label': Class labels (0 or 1) indicating which part of the circles the points belong to.
    
#     The dataset is saved to a CSV file named 'custom_concentric_circles_dataset.csv'.
#     """
    
#     # Define angle ranges for the segments of the circles
#     angles_right = np.linspace(-np.pi/2, np.pi/2, num_points // 2, endpoint=False)
#     angles_left = np.linspace(np.pi/2, 3*np.pi/2, num_points // 2, endpoint=False)

#     # Generate points for the smaller circle
#     x_right_small = radius_1 * np.cos(angles_right) + np.random.normal(0, noise_std, num_points // 2)
#     y_right_small = radius_1 * np.sin(angles_right) + np.random.normal(0, noise_std, num_points // 2)
#     x_left_small = radius_1 * np.cos(angles_left) + np.random.normal(0, noise_std, num_points // 2)
#     y_left_small = radius_1 * np.sin(angles_left) + np.random.normal(0, noise_std, num_points // 2)

#     # Generate points for the larger circle
#     x_right_large = radius_2 * np.cos(angles_right) + np.random.normal(0, noise_std, num_points // 2)
#     y_right_large = radius_2 * np.sin(angles_right) + np.random.normal(0, noise_std, num_points // 2)
#     x_left_large = radius_2 * np.cos(angles_left) + np.random.normal(0, noise_std, num_points // 2)
#     y_left_large = radius_2 * np.sin(angles_left) + np.random.normal(0, noise_std, num_points // 2)

#     # Combine the data
#     X = np.concatenate([x_right_small, x_left_large, x_right_large, x_left_small])
#     Y = np.concatenate([y_right_small, y_left_large, y_right_large, y_left_small])

#     # Assign labels
#     # Label 0: right part of smaller circle and left part of larger circle
#     # Label 1: right part of larger circle and left part of smaller circle
#     labels = np.concatenate([
#         np.zeros(num_points // 2),  # Label 0: right part of smaller circle
#         np.zeros(num_points // 2),  # Label 0: left part of larger circle
#         np.ones(num_points // 2),   # Label 1: right part of larger circle
#         np.ones(num_points // 2)    # Label 1: left part of smaller circle
#     ])

#     # Create a DataFrame
#     df = pd.DataFrame({
#         'X': X,
#         'Y': Y,
#         'Label': labels
#     })

#     # Save dataset to a CSV file
#     df.to_csv('custom_concentric_circles_dataset.csv', index=False)
    
#     return df

# def sample_pair_wise_data(df,num_samples=50):
#     """
#     Samples pairwise data from a given DataFrame.

#     Parameters:
#     - num_samples (int): Number of samples to draw from each label group. Default is 50.
#     - df (pd.DataFrame): Input DataFrame containing data with columns 'X', 'Y', and 'Label'.
#       The DataFrame should include:
#       - 'X': X-coordinates of the data points.
#       - 'Y': Y-coordinates of the data points.
#       - 'Label': Class labels (0 or 1) indicating the circle segments.

#     Returns:
#     - pd.DataFrame: A DataFrame containing paired sample data with columns:
#       - 'X1': X-coordinates of the sampled points from Label 0.
#       - 'Y1': Y-coordinates of the sampled points from Label 0.
#       - 'X2': X-coordinates of the sampled points from Label 1.
#       - 'Y2': Y-coordinates of the sampled points from Label 1.
#       - 'label': A column with a constant value of -1 (optional).
    
#     The sampled data is also saved to a CSV file named 'sampled_concentric_circles_dataset.csv'.
#     """
    
#     # Separate the dataset by label
#     df_label_0 = df[df['Label'] == 0]
#     df_label_1 = df[df['Label'] == 1]

#     # Sample points from each label group
#     sampled_label_0 = df_label_0.sample(n=num_samples, replace=False, random_state=42)  # Random sample from Label 0
#     sampled_label_1 = df_label_1.sample(n=num_samples, replace=False, random_state=42)  # Random sample from Label 1

#     # Extract coordinates
#     X1, Y1 = sampled_label_0['X'].values, sampled_label_0['Y'].values
#     X2, Y2 = sampled_label_1['X'].values, sampled_label_1['Y'].values

#     # Create a DataFrame for the sampled data
#     sampled_df = pd.DataFrame({
#         'X1': X1,
#         'Y1': Y1,
#         'X2': X2,
#         'Y2': Y2
#     })
    
#     # Add a constant label column with value -1
#     sampled_df['label'] = -1

#     # Save the sampled data to a CSV file
#     sampled_df.to_csv('sampled_concentric_circles_dataset.csv', index=False)
    
#     return sampled_df,sampled_label_0,sampled_label_1
def generate_sample_pair_wise_data(num_points=5000, radius_1=5, radius_2=10, noise_std=0.4,num_samples=50):
    """
    Samples pairwise data from a given DataFrame.

    Parameters:
    - num_samples (int): Number of samples to draw from each label group. Default is 50.
    - df (pd.DataFrame): Input DataFrame containing data with columns 'X', 'Y', and 'Label'.
      The DataFrame should include:
      - 'X': X-coordinates of the data points.
      - 'Y': Y-coordinates of the data points.
      - 'Label': Class labels (0 or 1) indicating the circle segments.

    Returns:
    - pd.DataFrame: A DataFrame containing paired sample data with columns:
      - 'X1': X-coordinates of the sampled points from Label 0.
      - 'Y1': Y-coordinates of the sampled points from Label 0.
      - 'X2': X-coordinates of the sampled points from Label 1.
      - 'Y2': Y-coordinates of the sampled points from Label 1.
      - 'label': A column with a constant value of -1 (optional).
    
    The sampled data is also saved to a CSV file named 'sampled_concentric_circles_dataset.csv'.
    """
    
    # Define angle ranges
    angles_right = np.linspace(-np.pi/2, np.pi/2, num_points // 2, endpoint=False)
    angles_left = np.linspace(np.pi/2, 3*np.pi/2, num_points // 2, endpoint=False)

    # Generate points for the smaller circle
    x_right_small = radius_1 * np.cos(angles_right) + np.random.normal(0, noise_std, num_points // 2)
    y_right_small = radius_1 * np.sin(angles_right) + np.random.normal(0, noise_std, num_points // 2)
    x_left_small = radius_1 * np.cos(angles_left) + np.random.normal(0, noise_std, num_points // 2)
    y_left_small = radius_1 * np.sin(angles_left) + np.random.normal(0, noise_std, num_points // 2)

    # Generate points for the larger circle
    x_right_large = radius_2 * np.cos(angles_right) + np.random.normal(0, noise_std, num_points // 2)
    y_right_large = radius_2 * np.sin(angles_right) + np.random.normal(0, noise_std, num_points // 2)
    x_left_large = radius_2 * np.cos(angles_left) + np.random.normal(0, noise_std, num_points // 2)
    y_left_large = radius_2 * np.sin(angles_left) + np.random.normal(0, noise_std, num_points // 2)

    #small and large data
    X_small=np.concatenate([x_right_small,x_left_small])
    X_large=np.concatenate([x_right_large,x_left_large])
    Y_small=np.concatenate([y_right_small,y_left_small])
    Y_large=np.concatenate([y_right_large,y_left_large])
    labels_small=np.concatenate([np.zeros(num_points // 2),np.ones(num_points // 2)])
    labels_large=np.concatenate([np.ones(num_points // 2),np.zeros(num_points // 2)])
    # Combine the data
    df_small=pd.DataFrame({'X':X_small,'Y':Y_small,'Label': labels_small})
    df_large=pd.DataFrame({'X':X_large,'Y':Y_large,'Label': labels_large})
    # Separate the dataset by label
    df_small_label_0 = df_small[df_small['Label'] == 0]
    df_small_label_1 = df_small[df_small['Label'] == 1]
    df_large_label_0 = df_large[df_large['Label'] == 0]
    df_large_label_1 = df_large[df_large['Label'] == 1]
    # df_label_0 = df[df['Label'] == 0]
    # df_label_1 = df[df['Label'] == 1]

    # Sample points from each label group
    sampled_small_label_0 = df_small_label_0.sample(n=num_samples//2, replace=False, random_state=42)  # Random sample from Label 0
    sampled_small_label_1 = df_small_label_1.sample(n=num_samples//2, replace=False, random_state=42)  # Random sample from Label 1
    sampled_large_label_0 = df_large_label_0.sample(n=num_samples//2, replace=False, random_state=42)  # Random sample from Label 0
    sampled_large_label_1 = df_large_label_1.sample(n=num_samples//2, replace=False, random_state=42)  # Random sample from Label 1
    # sampled_label_0 = df_label_0.sample(n=num_samples, replace=False, random_state=42)  # Random sample from Label 0
    # sampled_label_1 = df_label_1.sample(n=num_samples, replace=False, random_state=42)  # Random sample from Label 1

    # Extract coordinates
    X1_small, Y1_small = sampled_small_label_0['X'].values, sampled_small_label_0['Y'].values
    X2_small, Y2_small = sampled_small_label_1['X'].values, sampled_small_label_1['Y'].values
    X1_large, Y1_large = sampled_large_label_0['X'].values, sampled_large_label_0['Y'].values
    X2_large, Y2_large = sampled_large_label_1['X'].values, sampled_large_label_1['Y'].values

    #X = np.concatenate([x_right_small, x_left_large, x_right_large, x_left_small])
    #Y = np.concatenate([y_right_small, y_left_large, y_right_large, y_left_small])

    # Assign labels
    # Label 0: right part of smaller circle and left part of larger circle
    # Label 1: right part of larger circle and left part of smaller circle

    labels_small=np.concatenate([np.zeros(num_points // 2),np.ones(num_points // 2)])
    labels_large=np.concatenate([np.ones(num_points // 2),np.zeros(num_points // 2)])

    sampled_df = pd.DataFrame({
        'X1': np.concatenate([X1_small,X1_large]),
        'Y1': np.concatenate([Y1_small,Y1_large]),
        'X2': np.concatenate([X2_small,X2_large]),
        'Y2': np.concatenate([Y2_small,Y2_large])
    })
    sampled_df['label']=-1
    
    l0=pd.concat([sampled_small_label_0, sampled_large_label_0], axis=0, ignore_index=True)
    l1=pd.concat([sampled_small_label_1, sampled_large_label_1], axis=0, ignore_index=True)
    return sampled_df,l0,l1
def split_dataframe(df, test_size=0.3, random_state=None):
    """
    Splits a DataFrame into training and testing datasets.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator (for reproducibility).
        
    Returns:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """
    # Assuming the last column is the label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
def rkhs_data_prepare(data_set,frac,kernel):
    #dataframe in the form of features1,features2,label
    file_path = os.path.join('.', 'data', 'processed_data', f'{data_set}.csv')
    sampled_df=pd.read_csv(file_path)
    #drops the lable 
    sampled_df.drop(columns=['label'],inplace=True)
    #in case we want to work with a fraction 
    sampled_df = sampled_df.sample(frac=frac/100, random_state=42)
    #data frame contain single features to map to RKHS and compute alpha representation
    file_path = os.path.join('.', 'data', 'processed_data', f'{data_set}_single.csv')
    df_combined=pd.read_csv(file_path)
    #converts to a list
    my_list = df_combined.values.tolist()
    #holds the x -> k(x,.)
    RKHS_list = []
    for d in my_list:
        RKHS_list.append(RKHS(kernel, [1.0], [np.array(d)]))
    
    
    o, v = gram_schmidt(RKHS_list)
    #holds the alpha representation for k(x,.)
    alpha_list = []
    for d in RKHS_list:
        v = alpha_reper(o, d)
        alpha_list.append(v)
        #print("length",len(v),len(my_list),sampled_df.shape)
    assert len(alpha_list)==len(my_list)
    ##########################
    # Initialize a list to hold updated rows
    A=my_list
    B=alpha_list
    new_rows = []
    M1=len(alpha_list[0])
    M = sampled_df.shape[1] // 2
    for index, row in sampled_df.iterrows():
        # Split the row into l1 and l2
        l1 = row[:M].tolist()
        l2 = row[M:].tolist()
    
        # Find corresponding indices in A and get ll1 and ll2 from B
        ll1 = B[A.index(l1)] if l1 in A else [None] * M1
        ll2 = B[A.index(l2)] if l2 in A else [None] * M1
    
        # Combine ll1 and ll2 and append to the new rows
        new_rows.append(ll1 + ll2)

    # Create the new DataFrame
    new_df = pd.DataFrame(new_rows, columns=[f'Feature_{i+1}' for i in range(2 * M1)])
    new_df['Label'] = -1
    return new_df

def rkhs_synthetic_data_prepare(num_points,radius_1,radius_2,noise_std,num_samples,kernel):
    sampled_df, l0, l1 = generate_sample_pair_wise_data(num_points=num_points, radius_1=radius_1, radius_2=radius_2, noise_std=noise_std,num_samples=num_samples)
    df_combined = pd.concat([l0, l1], axis=0, ignore_index=True)
    
    my_list = df_combined[["X", "Y"]].values.tolist()
    RKHS_list = []
    for d in my_list:
        RKHS_list.append(RKHS(kernel, [1.0], [np.array(d)]))
    
    o, v = gram_schmidt(RKHS_list)
    alpha_list = []
    for d in RKHS_list:
        v = alpha_reper(o, d)
        alpha_list.append(v)    
    df = pd.DataFrame(alpha_list)
    df['Label'] = df_combined['Label']
    df1 = df[df['Label'] == 0.0].reset_index().drop('Label', axis=1)
    df2 = df[df['Label'] == 1.0].reset_index().drop('Label', axis=1)
    combined_df = pd.concat([df1, df2], axis=1)
    combined_df['Label'] = -1
    combined_df = combined_df.drop('index', axis=1)
    sampled_df = combined_df

    return sampled_df

def rkhs_data_prepare_2(data_set,frac,kernel,test_size):
    #dataframe in the form of features1,features2,label
    file_path = os.path.join('.', 'data', 'processed_data', f'{data_set}.csv')
    sampled_df=pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = split_dataframe(sampled_df, test_size, random_state=None)
    #drops the lable 
    sampled_df.drop(columns=['label'],inplace=True)
    #in case we want to work with a fraction 
    sampled_df = sampled_df.sample(frac=frac/100, random_state=42)
    #data frame contain single features to map to RKHS and compute alpha representation
    file_path = os.path.join('.', 'data', 'processed_data', f'{data_set}_single.csv')
    df_combined=pd.read_csv(file_path)
    #converts to a list
    #my_list = df_combined.values.tolist()
    my_list=X_train
    #holds the x -> k(x,.)
    RKHS_list = []
    for d in my_list:
        RKHS_list.append(RKHS(kernel, [1.0], [np.array(d)]))
    
    
    o, v = gram_schmidt(RKHS_list)
    #holds the alpha representation for k(x,.)
    alpha_list = []
    for d in RKHS_list:
        v = alpha_reper(o, d)
        alpha_list.append(v)
        #print("length",len(v),len(my_list),sampled_df.shape)
    assert len(alpha_list)==len(my_list)
    ##########################
    # Initialize a list to hold updated rows
    A=my_list
    B=alpha_list
    new_rows = []
    M1=len(alpha_list[0])
    M = sampled_df.shape[1] // 2
    for index, row in sampled_df.iterrows():
        # Split the row into l1 and l2
        l1 = row[:M].tolist()
        l2 = row[M:].tolist()
    
        # Find corresponding indices in A and get ll1 and ll2 from B
        ll1 = B[A.index(l1)] if l1 in A else [None] * M1
        ll2 = B[A.index(l2)] if l2 in A else [None] * M1
    
        # Combine ll1 and ll2 and append to the new rows
        new_rows.append(ll1 + ll2)

    # Create the new DataFrame
    new_df = pd.DataFrame(new_rows, columns=[f'Feature_{i+1}' for i in range(2 * M1)])
    new_df['Label'] = -1
    return new_df


