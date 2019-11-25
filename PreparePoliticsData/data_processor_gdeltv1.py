import data_handler as dh
import settings
import os

valid_countries = ['CHN', 'DEU', 'FRA', 'GBR', 'USA']

#dh.filter_file_gdeltv1(settings.GDELT1_SCHEMA, 'gdeltv1.TXT', 'events_gdeltv1.csv', valid_countries, 'events')

base_url = "http://data.gdeltproject.org/events/"
valid_countries = ['CHN', 'DEU', 'FRA', 'GBR', 'USA', 'CN', 'FR', 'DE', 'GB', 'US']
i = 1
part = 8
num_lines = sum(1 for line in open('../Res/exports_all.txt'))
file = open('../Res/exports_all.txt', mode='r')
for line in file:
    line = line.rstrip()
    try:
        downloaded_file = dh.download_file(base_url, line)
    except:
        continue

    extracted_file = dh.extract_file(downloaded_file)
    #if '.csv' not in extracted_file:
    #    extracted_file = extracted_file + '.csv'
    dh.filter_file(settings.GDELT2_SCHEMA, extracted_file, f'exports_all_part{part}.csv', valid_countries, 'events')
    dh.delete_file(downloaded_file)
    dh.delete_file(extracted_file)

    if os.path.getsize(f"../Res/exports_all_part{part}.csv") > 1000000000:
        part += 1

    print(f'\n{i}. iteration ended.\n')
    print(f'\n{(100 * i) / num_lines}% done.\n')
    i += 1

