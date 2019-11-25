import os.path
import urllib
import zipfile
import csv


def download_file(base_url, link):
    print('downloading {}'.format(link))
    filename = link.replace(base_url, '')
    urllib.request.urlretrieve(url=base_url + link, filename='../Res/' + filename)

    return filename


def extract_file(filename):
    print('extracting {}'.format(filename))
    z = zipfile.ZipFile(file='../Res/' + filename, mode='r')
    z.extractall(path='../Res/')

    return filename.replace('.zip', '')


def delete_file(filename):
    print('deleting {}'.format(filename))
    os.unlink('../Res/' + filename)


def filter_file(schema, to_filter_filename, filtered_filename, valid_countries, table_name):
    print('filtering {}'.format(to_filter_filename))
    dtypes = schema[table_name]['columns-dtypes']
    fieldnames = dtypes.keys()
    filtered_fieldnames = schema[table_name]['useful_cols']
    with open('../Res/' + filtered_filename, mode='a', encoding="utf8") as csv_filtered_file:
        csv_writer = csv.DictWriter(csv_filtered_file, fieldnames=filtered_fieldnames, delimiter='\t')
        with open('../Res/' + to_filter_filename, mode='r', encoding="utf8") as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames, delimiter='\t')
            for row in csv_reader:
                if is_row_matching(schema, row, valid_countries, table_name):
                    csv_writer.writerow(create_writer_dict(schema, row, table_name))


def is_row_matching(schema, row, valid_countries, table_name):
    if row['eventrootcode'] not in schema[table_name]['cameo-families']:
        return False

    for match in schema[table_name]['matches']:
        for country in valid_countries:
            if row[match] == country:
                return True
    return False


def filter_file_gdeltv1(schema, to_filter_filename, filtered_filename, valid_countries, table_name):
    print('filtering {}'.format(to_filter_filename))
    dtypes = schema[table_name]['columns-dtypes']
    fieldnames = list(dtypes.keys())
    filtered_fieldnames = schema[table_name]['useful_cols']

    i = 1
    num_lines = sum(1 for line in open(f'../Res/{to_filter_filename}') if 'Date' in line or int(line[:4]) > 1989 or int(line[:8]) < 20150219) - 1

    with open('../Res/' + filtered_filename, mode='a', encoding="utf8") as csv_filtered_file:
        csv_writer = csv.DictWriter(csv_filtered_file, fieldnames=filtered_fieldnames, delimiter='\t')
        with open('../Res/' + to_filter_filename, mode='r', encoding="utf8") as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames, delimiter='\t')
            next(csv_reader, None)
            for row in csv_reader:
                if int(row['Date'][:4]) < 1990:
                    continue
                if int(row['Date'][:8]) >= 20150219:
                    break

                if is_row_matching_gdeltv1(schema, row, valid_countries, table_name):
                    csv_writer.writerow(create_writer_dict(schema, row, table_name))
                print(f'\n{i}. iteration ended.\n')
                print(f'\n{(100 * i) / num_lines}% done.\n')
                i += 1


def is_row_matching_gdeltv1(schema, row, valid_countries, table_name):
    cameo_code = row['CAMEOCode']
    if cameo_code.startswith('0'):
        cameo_code = cameo_code[1]
    else:
        cameo_code = cameo_code[:2]

    if cameo_code not in schema[table_name]['cameo-families']:
        return False

    for match in schema[table_name]['matches']:
        for country in valid_countries:
            if country in row[match]:
                return True
    return False


def create_writer_dict(schema, row, table_name):
    writer_dict = {}
    for key in schema[table_name]['useful_cols']:
        writer_dict[key] = row[key]
    return writer_dict
