
def add_column_label(input_file: str, column_name: str):
    """
    input_file: str: Path to the input file
    column_name: str: Name of the column to be added
    """
    with open(input_file, 'w') as file:
        file.write(column_name+',')
        file.write(file.read())


# Call the clean_csv function
clean_csv(input_file, output_file, column_name)