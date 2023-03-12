import glob
import re
import os

def process_module(module_name, current_path, src_list):
    library_name = 'dataheroes'
    processed_module_name = module_name
    # replace module starts with "." according to current path
    if processed_module_name.strip() == '.':
        current_path_list = os.path.dirname(current_path).split('/')
        processed_module_name = '.'.join(current_path_list)

    if processed_module_name.strip().startswith('...'):
        current_path_list = os.path.dirname(current_path).split('/')
        prev_path_list = current_path_list[:-2]
        prev_path = '.'.join(prev_path_list)
        processed_module_name = prev_path +processed_module_name.replace('...','.')

    if processed_module_name.strip().startswith('..'):
        current_path_list = os.path.dirname(current_path).split('/')
        prev_path_list = current_path_list[:-1]
        prev_path = '.'.join(prev_path_list)
        processed_module_name = prev_path +processed_module_name.replace('..','.')
    #replace module starts with "." according to current path
    if processed_module_name.strip().startswith('.'):
        processed_module_name = os.path.dirname(current_path).replace('/','.')+processed_module_name
    #if module starts with skcoreset look in src dictionary and if find - replace last section of module
    # skcoreset.some.module => skcoreset.some.src$123
    if processed_module_name.strip().startswith(library_name):
        module_path = processed_module_name.replace('.','/').strip()+'.py'
        matches_in_src = [x for x in src_list if x.get('path') == module_path]
        if len(matches_in_src) > 0:
            old_module_name = processed_module_name.split('.')[-1]
            new_module_name = matches_in_src[0].get('new_path').replace(".py","").split('/')[-1]
            processed_module_name = processed_module_name.replace(old_module_name, new_module_name)
    print(f'{module_name=}, {processed_module_name=} ')
    return processed_module_name

def process_src():
    list_pyx = [
        "dataheroes/**/*.py",
        "dataheroes/*.py",
    ]

    src_list = []
    for file_pattern in list_pyx:
        for path in glob.glob(file_pattern,  recursive=True):
            with open(path) as file:
                src_file = file.readlines()

            with open(path, 'w') as file:
                for line in src_file:

                    if line.strip().startswith("@"):
                        pass
                    elif line.startswith("from"):
                        module_name = re.search('from(.*)import', line).group(1).strip()
                        if module_name.startswith('.'):
                            processed_module_name = process_module(module_name, path, src_list)
                        else:
                            processed_module_name = module_name
                        processed_line = line.replace(module_name, processed_module_name)
                        file.writelines(processed_line)
                    else:
                        file.writelines(line)
