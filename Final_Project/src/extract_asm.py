# import pandas as pd
# import os
# from collections import Counter

# # The standard x86 instructions we want to count
# OPCODES = [
#     'mov', 'push', 'pop', 'inc', 'dec', 'add', 'sub', 
#     'jmp', 'cmp', 'xor', 'call', 'ret', 'test', 'lea'
# ]

# def extract_opcodes(file_path):
#     """Reads an .asm file line-by-line and counts specific opcodes."""
#     opcode_counts = Counter()
#     try:
#         # latin-1 encoding is required because malware contains non-standard bytes
#         with open(file_path, 'r', encoding='latin-1') as f:
#             for line in f:
#                 parts = line.split()
#                 for part in parts:
#                     if part in OPCODES:
#                         opcode_counts[part] += 1
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#     return opcode_counts

# def main():
#     print("Starting opcode extraction...")
    
#     # 1. Load the list of 1,089 files
#     df = pd.read_csv('subset_train_list.csv')
#     data_dir = 'data/raw_sample/'
    
#     features_list = []
#     total_files = len(df)
    
#     # 2. Iterate through each file
#     for index, row in df.iterrows():
#         file_id = row['Id']
#         label = row['Class']
#         asm_path = os.path.join(data_dir, f"{file_id}.asm")
        
#         # Print a progress update every 100 files
#         if index % 100 == 0:
#             print(f"Processing file {index} of {total_files}...")
            
#         if os.path.exists(asm_path):
#             counts = extract_opcodes(asm_path)
            
#             # Start building the row dictionary
#             row_features = {'Id': file_id, 'Class': label}
            
#             # Add the count for every opcode
#             for opcode in OPCODES:
#                 row_features[opcode] = counts.get(opcode, 0)
                
#             features_list.append(row_features)
#         else:
#             print(f"Warning: File not found -> {asm_path}")
            
#     # 3. Convert all counts into a DataFrame and save
#     features_df = pd.DataFrame(features_list)
#     features_df.to_csv('data/asm_features.csv', index=False)
    
#     print("\nExtraction complete! Saved to data/asm_features.csv")
#     print("\nPreview of your new ML-ready dataset:")
#     print(features_df.head())

# if __name__ == '__main__':
#     main()

import pandas as pd
import os
from collections import Counter

OPCODES = [
    'mov', 'push', 'pop', 'inc', 'dec', 'add', 'sub', 
    'jmp', 'cmp', 'xor', 'call', 'ret', 'test', 'lea'
]

def extract_opcodes(file_path):
    opcode_counts = Counter()
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.split()
                for part in parts:
                    if part in OPCODES:
                        opcode_counts[part] += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return opcode_counts

def main():
    print("Starting opcode extraction...")
    
    # --- BULLETPROOF PATHING ---
    # 1. Find exactly where this script lives (.../finalproj/src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Step up one level to the main project root (.../finalproj/)
    project_root = os.path.dirname(script_dir)
    
    # 3. Define absolute paths based on the root
    list_path = os.path.join(project_root, 'subset_train_list.csv')
    data_dir = os.path.join(project_root, 'data', 'raw_sample')
    output_path = os.path.join(project_root, 'data', 'asm_features.csv')
    
    # Ensure the output 'data' directory actually exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # ---------------------------

    # Load the subset list
    df = pd.read_csv(list_path)
    features_list = []
    total_files = len(df)
    
    for index, row in df.iterrows():
        file_id = row['Id']
        label = row['Class']
        asm_path = os.path.join(data_dir, f"{file_id}.asm")
        
        if index % 100 == 0:
            print(f"Processing file {index} of {total_files}...")
            
        if os.path.exists(asm_path):
            counts = extract_opcodes(asm_path)
            
            row_features = {'Id': file_id, 'Class': label}
            for opcode in OPCODES:
                row_features[opcode] = counts.get(opcode, 0)
                
            features_list.append(row_features)
        else:
            print(f"Warning: File not found -> {asm_path}")
            
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_path, index=False)
    
    print(f"\nExtraction complete! Saved to {output_path}")
    print("\nPreview of your new ML-ready dataset:")
    print(features_df.head())

if __name__ == '__main__':
    main()
