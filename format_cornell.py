import csv
import codecs
import chatbot_utils as utils
import os

INFILE = "movie_lines.txt"
OUTFILE = "formatted_cornell.txt"

LIB_FOLDER = "Data/"

FIRST_DATA_COL = 8

print("Reading input file...")

raw_lines = codecs.open(INFILE, encoding='utf-8', errors='ignore'). \
        read().strip().split('\n')

lines=[]
for line in raw_lines:
    stripped_line = " ".join(line.split(" ")[FIRST_DATA_COL:])
    lines.append(utils.normalize_string(stripped_line).split(" "))

print("File read.")
print("Writing output...")
with open(LIB_FOLDER+OUTFILE, 'w') as out:
    writer = csv.writer(out)
    writer.writerows(lines)
print("Output written.")