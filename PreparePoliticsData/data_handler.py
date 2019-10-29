import os.path
import urllib
import zipfile
import csv
import settings


def download_file(link):
    base_url = "http://data.gdeltproject.org/gdeltv2/"
    print('downloading {}'.format(link))
    filename = link.replace(base_url, '')
    urllib.request.urlretrieve(url=link, filename='../Res/' + filename)

    return filename


def extract_file(filename):
    print('extracting {}'.format(filename))
    z = zipfile.ZipFile(file='../Res/' + filename, mode='r')
    z.extractall(path='../Res/')

    return filename.replace('.zip', '')


def delete_file(filename):
    print('deleting {}'.format(filename))
    os.unlink('../Res/' + filename)


def filter_file(to_filter_filename, filtered_filename, valid_countries, table_name):
    print('filtering {}'.format(to_filter_filename))
    dtypes = settings.SCHEMA[table_name]['columns-dtypes']
    fieldnames = dtypes.keys()
    filtered_fieldnames = settings.SCHEMA[table_name]['useful_cols']

    with open('../Res/' + filtered_filename, mode='a', encoding="utf8") as csv_filtered_file:
        csv_writer = csv.DictWriter(csv_filtered_file, fieldnames=filtered_fieldnames, delimiter='\t')
        with open('../Res/' + to_filter_filename, mode='r', encoding="utf8") as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames, delimiter='\t')
            for row in csv_reader:
                if is_row_matching(row, valid_countries, table_name):
                    csv_writer.writerow(create_writer_dict(row, table_name))


def is_row_matching(row, valid_countries, table_name):
    avg_tone = abs(float(row['avgtone']))
    if avg_tone < 4.5:
        return False

    if row['eventrootcode'] not in settings.SCHEMA[table_name]['cameo-families']:
        return False

    for match in settings.SCHEMA[table_name]['matches']:
        for country in valid_countries:
            if row[match] == country:
                return True
    return False


def create_writer_dict(row, table_name):
    writer_dict = {}
    for key in settings.SCHEMA[table_name]['useful_cols']:
        writer_dict[key] = row[key]
    return writer_dict
