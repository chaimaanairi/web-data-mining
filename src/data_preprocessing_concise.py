import pandas as pd

# Step 1: Load the original large dataset in chunks
chunk_size = 100000  # Adjust the chunk size as needed for your system's memory
chunks = pd.read_csv('E:/web-data-mining/results/preprocessed_data.csv', chunksize=chunk_size)

# Step 2: Create an empty DataFrame to hold the sampled rows
sampled_df = pd.DataFrame()

# Step 3: Process each chunk, sampling 20,000 rows from it
for chunk in chunks:
    sampled_df = pd.concat([sampled_df, chunk.sample(n=20000, random_state=42)])

# Step 4: Save the sampled data to a new CSV file
sampled_df.to_csv('E:/web-data-mining/results/preprocessed_data_concise.csv', index=False)

print("Concise dataset created successfully with 20,000 rows!")
