import pandas as pd
import os

def main():
    print("Starting N-Gram extraction. This might take a minute or two...")
    
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    features_path = os.path.join(project_root, 'data', 'features_with_metadata.csv')
    data_dir = os.path.join(project_root, 'data', 'raw_sample')
    output_path = os.path.join(project_root, 'data', 'features_final.csv')

    # Load existing upgraded features
    df = pd.read_csv(features_path)
    
    # Target 2-Grams
    target_ngrams = ['push_call', 'mov_mov', 'xor_xor', 'cmp_jmp', 'pop_ret']
    
    # Initialize lists to hold the counts
    ngram_counts = {ngram: [] for ngram in target_ngrams}
    valid_opcodes = {'push', 'call', 'mov', 'xor', 'cmp', 'jmp', 'pop', 'ret'}

    # 2. Iterate through files
    for index, row in df.iterrows():
        file_id = row['Id']
        asm_path = os.path.join(data_dir, f"{file_id}.asm")
        
        counts = {ngram: 0 for ngram in target_ngrams}
        
        if os.path.exists(asm_path):
            try:
                with open(asm_path, 'r', encoding='latin-1') as f:
                    prev_opcode = None
                    for line in f:
                        parts = line.split()
                        if len(parts) > 1:
                            # Opcode is usually the second string on the line
                            opcode = parts[1].lower()
                            if opcode in valid_opcodes:
                                if prev_opcode:
                                    bigram = f"{prev_opcode}_{opcode}"
                                    if bigram in target_ngrams:
                                        counts[bigram] += 1
                                prev_opcode = opcode
            except Exception as e:
                pass # Skip lines/files with severe encoding corruption
                
        for ngram in target_ngrams:
            ngram_counts[ngram].append(counts[ngram])

    # 3. Add to DataFrame
    for ngram in target_ngrams:
        df[ngram] = ngram_counts[ngram]
    
    # 4. Save
    df.to_csv(output_path, index=False)
    
    print("\nExtraction complete!")
    print(f"Saved final dataset to: {output_path}")

if __name__ == '__main__':
    main()