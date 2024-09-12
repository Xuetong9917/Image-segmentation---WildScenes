import random

def select_random_lines(input_file, output_file, num_lines=100):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Ensure num_lines does not exceed the total number of lines in the file
    num_lines = min(num_lines, len(lines))

    # Randomly select num_lines from the list of lines
    selected_lines = random.sample(lines, num_lines)

    # Write the selected lines to the output file
    with open(output_file, 'w') as file:
        for line in selected_lines:
            file.write(line)

input_file = 'train_full.txt'
output_file = 'train.txt'
select_random_lines(input_file, output_file, 5000)
