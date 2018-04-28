from pyrouge import Rouge155

if __name__ == "__main__":

    rouge = Rouge155()
    rouge.system_dir = 'path/to/system_summaries'
    rouge.model_dir = 'path/to/model_summaries'
    rouge.system_filename_pattern = 'some_name.(\d+).txt'
    rouge.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'

    output = rouge.convert_and_evaluate()
    print(output)
    output_dict = rouge.output_to_dict(output)