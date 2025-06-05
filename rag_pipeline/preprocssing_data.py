from langchain_community.document_loaders import CSVLoader
import argparse
from pathlib import Path


def read_csv_file(file_path):
    """
    load given csv file
    """

    ipl_2022_data = CSVLoader(Path(file_path).resolve())
    ipl_2022_csv = ipl_2022_data.load()

    return ipl_2022_csv


def generate_player_summary(player_data_dict):
    """
    generate summary for given stats
    """
    
    return f"""
{player_data_dict['Player']} played for the {player_data_dict['IPl_team']} in {player_data_dict['year']} as a {player_data_dict['Role']}.
He played {player_data_dict['Matches']} matches and batted in {player_data_dict['Innings']} innings with {player_data_dict['Not_out']} not out(s).
He scored a total of {player_data_dict['Runs']} runs, with a highest score of {player_data_dict['Highest_Score']} and an average of {player_data_dict['Average']}.
He faced {player_data_dict['Balls_faced']} balls with a strike rate of {player_data_dict['Strike_rate']}.
He hit {player_data_dict["4's"]} fours and {player_data_dict["6's"]} sixes, including {player_data_dict['Centuries']} centuries and {player_data_dict['Half_centuaries']} half-centuries.
"""


def parse_the_csv(csv_file):
    """
    parse the raw stats into readable format so that the LLM can under true meaning of the stats
    """

    # empty string to store all the stats
    player_stat_summary=""

    # parsing each item into reading string
    for data in csv_file:
        player_data_dict = {}
        player_data = data.page_content
        player_data_split = player_data.split('\n')
        player_data_dict = {key.strip(): value.strip() for stat in player_data_split for key, value in [stat.split(':')]}

        player_summary = generate_player_summary(player_data_dict)
        player_stat_summary += player_summary + "\n\n"

    return player_stat_summary


def store_string_as_text_file(string, des_path):
    """
    storing string into a text file
    """

    with open(file=Path(des_path).resolve(), mode='w',encoding='utf-8') as file:
        file.write(string)


def main(args):

    # check the input file exist or not
    if not Path(args.input_csv_file_path).resolve().exists():
        print(f"Error: File not found -> {args.input_csv_file_path}")
        return 1

    # read the csv
    csv_data = read_csv_file(args.input_csv_file_path)
    if len(csv_data)==0:
        print("Given CSV has 0 data")
        return 1
    else:
        print("csv read sucessfully")

    # parse each player's data
    try:
        parsed_data = parse_the_csv(csv_data)
    except Exception as E:
        print(f"Error will parsing each player's stats")
        return 1

    # the parsed data into text file
    store_string_as_text_file(parsed_data, args.output_text_file_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Args Parser for data processing")
    args_parser.add_argument("--input_csv_file_path", default=".\..\data\IPL_2022.csv")
    args_parser.add_argument("--output_text_file_path", default=".\..\data\IPL_2022_summary.txt")

    # parsing arguments
    args = args_parser.parse_args()

    # calling main function
    main(args)