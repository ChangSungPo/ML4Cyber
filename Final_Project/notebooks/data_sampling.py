import pandas as pd
import numpy as np

# Load your labels
df = pd.read_csv('trainLabels.csv')

def get_stratified_sample(df, sample_pct=0.1, min_samples=10):
    """
    Selects a percentage of samples from each class, 
    ensuring a minimum count for small classes.
    """
    sampled_dfs = []
    
    for class_id in df['Class'].unique():
        class_subset = df[df['Class'] == class_id]
        n_samples = max(min_samples, int(len(class_subset) * sample_pct))
        
        # If the class is already smaller than min_samples, take them all
        if len(class_subset) <= n_samples:
            sampled_dfs.append(class_subset)
        else:
            sampled_dfs.append(class_subset.sample(n=n_samples, random_state=42))
            
    return pd.concat(sampled_dfs).reset_index(drop=True)

# Generate the subset list
subset_df = get_stratified_sample(df)

print(f"Original Dataset Size: {len(df)}")
print(f"Subset Dataset Size: {len(subset_df)}")
print("\n--- New Class Distribution ---")
print(subset_df['Class'].value_counts())

# Save this list so you know which files to unzip/process
subset_df.to_csv('subset_train_list.csv', index=False)
