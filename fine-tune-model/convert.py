import csv

def convert_csv_to_output(input_file, output_file):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the first row

        with open(output_file, 'w') as outfile:
            for row in reader:
                column_1 = row[0]
                column_2 = row[1]
                outfile.write(f"<SFT><s>[INST] <<SYS>\n You are a high school science teacher. Explain concepts in very simple English. \n<</SYS>>\n\n {column_1} [/INST] {column_2} \n")

# Testing the function
input_file = 'input.csv'  # Replace with your input file name
output_file = 'output.txt'  # Replace with your desired output file name

convert_csv_to_output(input_file, output_file)
