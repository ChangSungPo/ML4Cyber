import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 1. Bulletproof pathing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'asm_features.csv')
    
    # 2. Create a folder for your report images
    figures_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading feature matrix...")
    df = pd.read_csv(data_path)

    print("Generating boxplot for 'jmp' instructions...")
    
    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    
    # We use a boxplot to show the median and spread of the data per class
    sns.boxplot(x='Class', y='jmp', data=df)
    
    # Using a log scale because instruction counts vary wildly (from 0 to 100,000+)
    plt.yscale('symlog') 
    
    plt.title('Control Flow: JMP Instructions per Malware Class')
    plt.ylabel('JMP Count (Log Scale)')
    plt.xlabel('Malware Family (Class)')
    
    # 4. Save the plot securely
    plot_path = os.path.join(figures_dir, 'jmp_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Success! Graph saved to: {plot_path}")

if __name__ == '__main__':
    main()
