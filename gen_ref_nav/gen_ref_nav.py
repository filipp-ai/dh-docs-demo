"""Generate the code reference pages and navigation."""
from pathlib import Path
import mkdocs_gen_files
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

                    if line.strip().startswith("@telemetry"):
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

process_src()
nav = mkdocs_gen_files.Nav()
for file_pattern in [
                    "dataheroes/services/coreset_tree/lg.py",
                    "dataheroes/services/coreset_tree/svd.py",
                    "dataheroes/services/coreset_tree/pca.py",
                     ]:
    for path_str in glob.glob(file_pattern, recursive=True):
        path = Path(path_str)
        module_path = path.relative_to("dataheroes").with_suffix("")
        doc_path = path.relative_to("dataheroes").with_suffix(".md")
        full_doc_path = Path("reference", doc_path)
        parts = tuple(module_path.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")
        mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

reference_nav_lines = []
for line in nav.build_literate_nav():
    line = line.replace('[lg]', '[CoresetTreeServiceLG]')
    line = line.replace('[svd]', '[CoresetTreeServiceSVD]')
    line = line.replace('[pca]', '[CoresetTreeServicePCA]')
    reference_nav_lines.append(line)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(reference_nav_lines)
# we do not use all this yet (but could in the future)
if False:
    src_path_list = [
        'examples/cleaning/examples/data_cleaning_object_detection_coco.ipynb',
        'examples/cleaning/examples/data_cleaning_coreset_vs_random_image_classification_cifar10.ipynb',
        'examples/cleaning/examples/data_cleaning_labeling_utility_image_classification_cifar10.ipynb',
        'examples/cleaning/examples/data_cleaning_image_classification_imagenet.ipynb',
        'examples/tree_service/build_options_tabular_data_covertype.ipynb',
        'examples/tree_service/all_library_functions_tabular_data_covertype.ipynb',
    ]

    for src_path in src_path_list:
        src_file_name = src_path.split('/')[-1]
        with mkdocs_gen_files.open(src_file_name, "w") as f:
            with open(src_path, 'r') as src:
                lines = src.readlines()
                for line in lines:
                    f.write(line)
