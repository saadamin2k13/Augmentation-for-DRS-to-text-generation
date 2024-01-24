# Read the dataset file
with open('../name-augmentation-files/randomly_sampled_05/drs_cleaned-01.txt', 'r') as f:
    dataset = f.read()

# Split the dataset into examples using the empty line separator
examples = dataset.strip().split('\n\n')

# Initialize a variable to store the cleaned dataset
cleaned_dataset = []

# Iterate through the examples
for example in examples:
    # Split the example into lines
    lines = example.split('\n')

    # Filter out lines containing "%" and remove content after "%" character
    cleaned_lines = []
    for line in lines:
        line = line.split('%', 1)[0].strip()  # Remove content after "%" character
        cleaned_lines.append(line)

    # Join the cleaned lines back into an example
    cleaned_example = '\n'.join(cleaned_lines)

    # Add the cleaned example to the cleaned dataset
    cleaned_dataset.append(cleaned_example)

# Join the cleaned examples with empty lines to preserve the structure
cleaned_data = '\n\n'.join(cleaned_dataset)

# Write the cleaned data to a new file
with open('../name-augmentation-files/randomly_sampled_05/drs_cleaned-02.txt', 'w') as f:
    f.write(cleaned_data)
