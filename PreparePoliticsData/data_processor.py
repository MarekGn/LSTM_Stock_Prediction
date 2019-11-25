import data_handler as dh
import settings

base_url = "http://data.gdeltproject.org/gdeltv2/"
valid_countries = ['CHN', 'DEU', 'FRA', 'GBR', 'USA', 'CN', 'FR', 'DE', 'GB', 'US']
i = 1
num_lines = sum(1 for line in open('../Res/exports.txt'))
file = open('../Res/exports.txt', mode='r')
for line in file:
    line = line.rstrip()
    try:
        downloaded_file = dh.download_file(base_url, line)
    except:
        continue

    extracted_file = dh.extract_file(downloaded_file)
    dh.filter_file(settings.GDELT2_SCHEMA, extracted_file, 'events_all.csv', valid_countries, 'events')
    dh.delete_file(downloaded_file)
    dh.delete_file(extracted_file)

    print(f'\n{i}. iteration ended.\n')
    print(f'\n{(100 * i) / num_lines}% done.\n')
    i += 1
