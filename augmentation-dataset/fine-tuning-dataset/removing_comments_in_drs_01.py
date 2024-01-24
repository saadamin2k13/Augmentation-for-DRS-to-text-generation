# Read the dataset file
with open('../name-augmentation-files/randomly_sampled_05/shuffled_drs.txt', 'r') as f:
    dataset = f.read()

# Split the dataset into examples using the empty line separator
examples = dataset.strip().split('\n\n')

# Initialize a variable to store the cleaned dataset
cleaned_dataset = []

# Iterate through the examples
for example in examples:
    # Split the example into lines
    lines = example.split('\n')

    # Filter out lines starting with "%"
    cleaned_lines = [line for line in lines if not line.strip().startswith('%')]

    # Join the cleaned lines back into an example
    cleaned_example = '\n'.join(cleaned_lines)

    # Add the cleaned example to the cleaned dataset
    cleaned_dataset.append(cleaned_example)

# Join the cleaned examples with empty lines to preserve the structure
cleaned_data = '\n\n'.join(cleaned_dataset)

# Write the cleaned data to a new file
with open('../name-augmentation-files/randomly_sampled_05/drs_cleaned-01.txt', 'w') as f:
    f.write(cleaned_data)
