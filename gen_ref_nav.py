"""Generate the code reference pages and navigation."""
import glob
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for file_pattern in sorted([
                    "dataheroes/services/coreset_tree/kmeans.py",
                    "dataheroes/services/coreset_tree/lg.py",
                    "dataheroes/services/coreset_tree/dtc.py",
                    "dataheroes/services/coreset_tree/dtr.py",
                    "dataheroes/services/coreset_tree/lr.py",
                    "dataheroes/services/coreset_tree/pca.py",
                    "dataheroes/services/coreset_tree/svd.py",
                    "dataheroes/data/common.py",
                     ]):
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

reference_nav_lines = []
for line in nav.build_literate_nav():
    line = line.replace('[lg]', '[CoresetTreeServiceLG]')
    line = line.replace('[dtc]', '[CoresetTreeServiceDTC]')
    line = line.replace('[dtr]', '[CoresetTreeServiceDTR]')
    line = line.replace('[svd]', '[CoresetTreeServiceSVD]')
    line = line.replace('[pca]', '[CoresetTreeServicePCA]')
    line = line.replace('[kmeans]', '[CoresetTreeServiceKMeans]')
    line = line.replace('[lr]', '[CoresetTreeServiceLR]')
    reference_nav_lines.append(line)

# put DataParams to end of content list (Code Reference) and without hierarchy
reference_nav_lines = reference_nav_lines[2:] + ['* [DataParams](data/common.md)']

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(reference_nav_lines)

# we do not use all this yet (but could in the future)
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
