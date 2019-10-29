def create_txt_files(filename, data):
    with open(f'{filename}.txt', 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line + '\n')


exports = []
mentions = []
gkgs = []

file = open(f'masterfilelist.txt', 'r', encoding='utf-8')
for line in file:
    line = line.replace('/n', '')
    line_parts = line.split()
    link = line_parts[-1]
    if "export" in link:
        exports.append(link)
    elif "mentions" in link:
        mentions.append(link)
    else:
        gkgs.append(link)
create_txt_files("exports", exports)
create_txt_files("mentions", mentions)
create_txt_files("gkgs", gkgs)
