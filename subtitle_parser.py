"""
Creates readable text file from SRT file.
"""
import re, sys


def clean_up(lines):
    """
    Get rid of all non-text lines and
    try to combine text broken into multiple lines
    """
    new_lines = []
    for i in range(len(lines[:-2])):
        if '-->' in lines[i + 1] or '-->' in lines[i]:
            continue
        new_lines.append(lines[i])
        print(lines[i])
    return new_lines


def main(args):
    """
      args[1]: file name
      args[2]: encoding. Default: utf-8.
        - If you get a lot of [?]s replacing characters,
        - you probably need to change file_encoding to 'cp1252'
    """
    file_name = args[1]
    # file_encoding = 'utf-8' if len(args) < 3 else args[2]
    with open(file_name, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        new_lines = clean_up(lines)
    new_file_name = file_name[:-4] + '.txt'
    with open(new_file_name, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)
    print("done")


if __name__ == '__main__':
    main(sys.argv)

"""
NOTES
 * Run from command line as
 ** python srt_to_txt.py file_name.srt cp1252
 * Creates file_name.txt with extracted text from file_name.srt 

 * Script assumes that lines beginning with lowercase letters or commas 
 * are part of the previous line and lines beginning with any other character
 * are new lines. This won't always be correct. 
"""
