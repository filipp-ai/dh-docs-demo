image_dataset_comment_1 = `
<span class=\\"commentText\\"># A prerequisite to use this Coreset is to extract the features embeddings from your images.
# In order to do so, drop the last classification layer from your pre-trained network, so the output would be the embeddings instead of the class distribution.
# To see how we extracted the feature embeddings from the ImageNet dataset using ResNet18 and PyTorch, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/feature_extraction_scripts/feature_extraction_imagenet1k_resnet18_pytorch.py\">link</a>.
# To see how we extracted the feature embeddings from the CIFAR10 dataset using ResNet18 and TensorFlow /Keras, visit this <a target=\"_blank\" href=https://github.com/Data-Heroes/dataheroes/blob/master/examples/feature_extraction_scripts/feature_extraction_cifar10_resnet18_tensorflow_keras.py">link</a>.
</span>
`

const algToClass = {
        "Linear Regression": 'CoresetTreeServiceLR',
        "Logistic Regression": 'CoresetTreeServiceLG',
        "K-Means": 'CoresetTreeServiceKMeans',
        "PCA": 'CoresetTreeServicePCA',
        "SVD": 'CoresetTreeServiceSVD',
        "Decision trees classification based": 'CoresetTreeServiceDTC',
        "Decision trees regression based": 'CoresetTreeServiceDTR',
        "Deep learning classification": 'CoresetTreeServiceLG',
        "Deep learning regression": 'CoresetTreeServiceLR',
    }

const unsupervisedAlg = ['K-Means', 'PCA', 'SVD'];

function genBuildFromFile(
            singleMultFilesDirs,
            targetFeaturesSeparate,
            fileTypeSelect,
            partial)
{
    let build_from_npy_file_single_separate = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.</span>
data_generated: Path = <span class=\"highlightText\">'generated'</span>
x_train_file = data_generated / <span class=\"highlightText\">'x_train.npy'</span>
y_train_file = data_generated / <span class=\"highlightText\">'y_train.npy'</span>

<span class=\"commentText\"># import reader for NPY format</span>
from dataheroes.data.utils import read_npy

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(x_train_file, y_train_file, reader_f=read_npy)
`

let partial_build_from_npy_file_single_separate = `
x_train_file_2 = data_generated / <span class=\"highlightText\">'x_train_2.npy'</span>
y_train_file_2 = data_generated / <span class=\"highlightText\">'y_train_2.npy'</span>
service_obj.partial_build_from_file(x_train_file_2, y_train_file_2, reader_f=read_npy)
`

let build_from_csv_file_single_separate = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span>
data_generated: Path = <span class=\"highlightText\">'generated'</span> 
x_train_file = data_generated / <span class=\"highlightText\">'x_train.csv'</span>
y_train_file = data_generated / <span class=\"highlightText\">'y_train.csv'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(x_train_file, y_train_file)
`

let partial_build_from_csv_file_single_separate = `
x_train_file_2 = data_generated / <span class=\"highlightText\">'x_train_2.csv'</span>
y_train_file_2 = data_generated / <span class=\"highlightText\">'y_train_2.csv'</span>
service_obj.partial_build_from_file(x_train_file_2, y_train_file_2)
`

let build_from_tsv_file_single_separate = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span>
data_generated: Path = <span class=\"highlightText\">'generated'</span> 
x_train_file = data_generated / <span class=\"highlightText\">'x_train.tsv'</span>
y_train_file = data_generated / <span class=\"highlightText\">'y_train.tsv'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(x_train_file, y_train_file, sep='\\t')
`

let partial_build_from_tsv_file_single_separate = `
x_train_file_2 = data_generated / <span class=\"highlightText\">'x_train_2.tsv'</span>
y_train_file_2 = data_generated / <span class=\"highlightText\">'y_train_2.tsv'</span>
service_obj.partial_build_from_file(x_train_file_2, y_train_file_2, sep='\\t')
`

let build_from_npy_file_single = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.</span>
data_generated: Path = <span class=\"highlightText\">'generated'</span>
data_file = data_generated / <span class=\"highlightText\">'data.npy'</span>

<span class=\"commentText\"># import reader for NPY format</span>
from dataheroes.data.utils import read_npy

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(data_file, reader_f=read_npy)
`

let partial_build_from_npy_file_single = `    
data_file_2 = data_generated / <span class=\"highlightText\">'data_2.npy'</span>
service_obj.partial_build_from_file(data_file_2, reader_f=read_npy)
`

let build_from_csv_file_single = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span> 
data_generated: Path = <span class=\"highlightText\">'generated'</span>
data_file = data_generated / <span class=\"highlightText\">'data.csv'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(data_file)
`

let partial_build_from_csv_file_single = `
data_file_2 = data_generated / <span class=\"highlightText\">'data_2.csv'</span>
service_obj.partial_build_from_file(data_file_2)
`

let build_from_tsv_file_single = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span> 
data_generated: Path = <span class=\"highlightText\">'generated'</span>
data_file = data_generated / <span class=\"highlightText\">'data.tsv'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(data_file, sep='\\t')
`

let partial_build_from_tsv_file_single = `
data_file_2 = data_generated / <span class=\"highlightText\">'data_2.tsv'</span>
service_obj.partial_build_from_file(data_file_2, sep='\\t')
`

let build_from_npy_file_mult = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.</span>
data_generated: Path = <span class=\"highlightText\">'generated'</span>
data_file1 = data_generated / <span class=\"highlightText\">'data1.npy'</span>
data_file2 = data_generated / <span class=\"highlightText\">'data2.npy'</span>

<span class=\"commentText\"># import reader for NPY format</span>
from dataheroes.data.utils import read_npy

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file([data_file1, data_file2], reader_f=read_npy)
`

let partial_build_from_npy_file_mult = `
data_file1_2 = data_generated / <span class=\"highlightText\">'data1_2.npy'</span>
data_file2_2 = data_generated / <span class=\"highlightText\">'data2_2.npy'</span>
service_obj.partial_build_from_file([data_file1_2, data_file2_2], reader_f=read_npy)
`

let build_from_csv_file_mult = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span> 
data_generated: Path = <span class=\"highlightText\">'generated'</span>
data_file1 = data_generated / <span class=\"highlightText\">'data1.npy'</span>
data_file2 = data_generated / <span class=\"highlightText\">'data2.npy'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file([data_file1, data_file2])
`

let partial_build_from_csv_file_mult = `
data_file1_2 = data_generated / <span class=\"highlightText\">'data1_2.npy'</span>
data_file2_2 = data_generated / <span class=\"highlightText\">'data2_2.npy'</span>
service_obj.partial_build_from_file([data_file1_2, data_file2_2])
`

let build_from_tsv_file_mult = `
data_generated: Path = <span class=\"highlightText\">'generated'</span>
data_file1 = data_generated / <span class=\"highlightText\">'data1.npy'</span>
data_file2 = data_generated / <span class=\"highlightText\">'data2.npy'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file([data_file1, data_file2], sep='\\t')
`

let partial_build_from_tsv_file_mult = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span> 
data_file1_2 = data_generated / <span class=\"highlightText\">'data1_2.npy'</span>
data_file2_2 = data_generated / <span class=\"highlightText\">'data2_2.npy'</span>
service_obj.partial_build_from_file([data_file1_2, data_file2_2], sep='\\t')
`

let build_from_npy_file_mult_separate = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.</span>
data_generated: Path = <span class=\"highlightText\">'generated'</span>
x_train_file1 = data_generated / <span class=\"highlightText\">'x_train1.npy'</span>
x_train_file2 = data_generated / <span class=\"highlightText\">'x_train2.npy'</span>
y_train_file1 = data_generated / <span class=\"highlightText\">'y_train1.npy'</span>
y_train_file2 = data_generated / <span class=\"highlightText\">'y_train2.npy'</span>

<span class=\"commentText\"># import reader for NPY format</span>
from dataheroes.data.utils import read_npy

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file([x_train_file1, x_train_file2], [y_train_file1, y_train_file2], reader_f=read_npy)
`

let partial_build_from_npy_file_mult_separate = `
x_train_file1_2 = data_generated / <span class=\"highlightText\">'x_train1_2.npy'</span>
x_train_file2_2 = data_generated / <span class=\"highlightText\">'x_train2_2.npy'</span>
y_train_file1_2 = data_generated / <span class=\"highlightText\">'y_train1_2.npy'</span>
y_train_file2_2 = data_generated / <span class=\"highlightText\">'y_train2_2.npy'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.partial_build_from_file([x_train_file1_2, x_train_file2_2], [y_train_file1_2, y_train_file2_2], reader_f=read_npy)
`

let build_from_csv_file_mult_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
data_generated: Path = <span class=\"highlightText\">'generated'</span>
x_train_file1 = data_generated / <span class=\"highlightText\">'x_train1.csv'</span>
x_train_file2 = data_generated / <span class=\"highlightText\">'x_train2.csv'</span>
y_train_file1 = data_generated / <span class=\"highlightText\">'y_train1.csv'</span>
y_train_file2 = data_generated / <span class=\"highlightText\">'y_train2.csv'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file([x_train_file1, x_train_file2], [y_train_file1, y_train_file2])
`

let partial_build_from_csv_file_mult_separate = `
x_train_file1_2 = data_generated / <span class=\"highlightText\">'x_train1_2.csv'</span>
x_train_file2_2 = data_generated / <span class=\"highlightText\">'x_train2_2.csv'</span>
y_train_file1_2 = data_generated / <span class=\"highlightText\">'y_train1_2.csv'</span>
y_train_file2_2 = data_generated / <span class=\"highlightText\">'y_train2_2.csv'</span>
service_obj.partial_build_from_file([x_train_file1_2, x_train_file2_2], [y_train_file1_2, y_train_file2_2])
`

let build_from_tsv_file_mult_separate = `
<span class=\"commentText\"># Prepare the definitions of path variables for the dataset files (replace with your directory name and file names)</span> 
data_generated: Path = <span class=\"highlightText\">'generated'</span>
x_train_file1 = data_generated / <span class=\"highlightText\">'x_train1.tsv'</span>
x_train_file2 = data_generated / <span class=\"highlightText\">'x_train2.tsv'</span>
y_train_file1 = data_generated / <span class=\"highlightText\">'y_train1.tsv'</span>
y_train_file2 = data_generated / <span class=\"highlightText\">'y_train2.tsv'</span>

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file([x_train_file1, x_train_file2], [y_train_file1, y_train_file2], sep='\\t')
`

let partial_build_from_tsv_file_mult_separate = `
x_train_file1_2 = data_generated / <span class=\"highlightText\">'x_train1_2.tsv'</span>
x_train_file2_2 = data_generated / <span class=\"highlightText\">'x_train2_2.tsv'</span>
y_train_file1_2 = data_generated / <span class=\"highlightText\">'y_train1_2.tsv'</span>
y_train_file2_2 = data_generated / <span class=\"highlightText\">'y_train2_2.tsv'</span>
service_obj.partial_build_from_file([x_train_file1_2, x_train_file2_2], [y_train_file1_2, y_train_file2_2], sep='\\t')
`

let build_from_dir_single = `
<span class=\"commentText\"># Prepare the data directory (replace with your directory name) and build the Coreset tree from the files in it.</span>
from pathlib import Path
data_dir = Path("<span class=\"highlightText\">data_dir</span>")

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(data_dir)
`

let partial_build_from_dir_single = `
data_dir_2 = Path("<span class=\"highlightText\">data_dir_2</span>")
service_obj.partial_build_from_file(data_dir_2)
`

let build_from_dir_single_separate = `
<span class=\"commentText\"># Prepare the data and target directories (replace with your directories names) 
# and build the Coreset tree from the files in it.</span>
from pathlib import Path
data_dir = Path("<span class=\"highlightText\">data_dir</span>")
target_dir = Path("<span class=\"highlightText\">target_dir</span>")

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(file_path=data_dir, target_file_path=target_dir)
`

let partial_build_from_dir_single_separate = `
data_dir_2 = Path("<span class=\"highlightText\">data_dir_2</span>")
target_dir_2 = Path("<span class=\"highlightText\">target_dir_2</span>")
service_obj.partial_build_from_file(file_path=data_dir_2, target_file_path=target_dir_2)
`

let build_from_dir_mult = `
<span class=\"commentText\"># Prepare the data and target directories (replace with your directories names) 
# and build the Coreset tree from the files in it.</span>
from pathlib import Path
data_dir1 = Path("<span class=\"highlightText\">data_dir1</span>")
data_dir2 = Path("<span class=\"highlightText\">data_dir2</span>")

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(file_path=[data_dir1, data_dir2]])
`

let partial_build_from_dir_mult = `
data_dir1_2 = Path("<span class=\"highlightText\">data_dir1_2</span>")
data_dir2_2 = Path("<span class=\"highlightText\">data_dir2_2</span>")

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.partial_build_from_file(file_path=[data_dir1_2, data_dir2_2]])
`

let build_from_dir_mult_separate = `
<span class=\"commentText\"># Prepare the data and target directories (replace with your directories names) 
# and build the Coreset tree from the files in it.</span>
from pathlib import Path
data_dir1 = Path("<span class=\"highlightText\">data_dir1</span>")
data_dir2 = Path("<span class=\"highlightText\">data_dir2</span>")
target_dir1 = Path("<span class=\"highlightText\">target_dir1</span>")
target_dir2 = Path("<span class=\"highlightText\">target_dir2</span>")

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.build_from_file(file_path=[data_dir1, data_dir2], target_file_path=[target_dir1, target_dir2])
`

let partial_build_from_dir_mult_separate = `
data_dir1_2 = Path("<span class=\"highlightText\">data_dir1_2</span>")
data_dir2_2 = Path("<span class=\"highlightText\">data_dir2_2</span>")
target_dir1_2 = Path("<span class=\"highlightText\">target_dir1_2</span>")
target_dir2_2 = Path("<span class=\"highlightText\">target_dir2_2</span>")

<span class=\"commentText\"># Build the Coreset Tree</span>
service_obj.partial_build_from_file(file_path=[data_dir1_2, data_dir2_2], target_file_path=[target_dir1_2, target_dir2_2])
`

    let codeSnippetText = partial ? `<span class=\"commentText\"># Add additional data to the Coreset tree</span>` : '';

    if (singleMultFilesDirs === 'Single Directory' && targetFeaturesSeparate === 'No') {
        codeSnippetText += partial ? partial_build_from_dir_single : build_from_dir_single;
    } else if (singleMultFilesDirs === 'Single Directory' && targetFeaturesSeparate === 'Yes') {
        codeSnippetText += partial ? partial_build_from_dir_single_separate : build_from_dir_single_separate
    } else if (singleMultFilesDirs === 'Multiple Directories' && targetFeaturesSeparate === 'No') {
        codeSnippetText += partial ? partial_build_from_dir_mult : build_from_dir_mult
    } else if (singleMultFilesDirs === 'Multiple Directories' && targetFeaturesSeparate === 'Yes') {
        codeSnippetText += partial ? partial_build_from_dir_mult_separate : build_from_dir_mult_separate
    } else if (singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'Yes' && fileTypeSelect == 'NPY') {
        codeSnippetText += partial ? partial_build_from_npy_file_single_separate : build_from_npy_file_single_separate;
    } else if (singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'Yes' && fileTypeSelect == 'TSV') {
        codeSnippetText += partial ? partial_build_from_tsv_file_single_separate : build_from_tsv_file_single_separate;
    } else if (singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'Yes' && fileTypeSelect == 'CSV') {
        codeSnippetText += partial ? partial_build_from_csv_file_single_separate : build_from_csv_file_single_separate;
    } else if (singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'No' && fileTypeSelect == 'NPY') {
        codeSnippetText += partial ? partial_build_from_npy_file_single : build_from_npy_file_single;
    } else if (singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'No' && fileTypeSelect == 'TSV') {
        codeSnippetText += partial ? partial_build_from_tsv_file_single : build_from_tsv_file_single;
    } else if (singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'No' && fileTypeSelect == 'CSV') {
        codeSnippetText += partial ? partial_build_from_csv_file_single : build_from_csv_file_single;
    } else if (singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'No' && fileTypeSelect == 'NPY') {
        codeSnippetText += partial ? partial_build_from_npy_file_mult : build_from_npy_file_mult;
    } else if (singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'No' && fileTypeSelect == 'TSV') {
        codeSnippetText += partial ? partial_build_from_tsv_file_mult : build_from_tsv_file_mult;
    } else if (singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'No' && fileTypeSelect == 'CSV') {
        codeSnippetText += partial ? partial_build_from_csv_file_mult : build_from_csv_file_mult;
    } else if (singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'Yes' && fileTypeSelect == 'NPY') {
        codeSnippetText += partial ? partial_build_from_npy_file_mult_separate : build_from_npy_file_mult_separate;
    } else if (singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'Yes' && fileTypeSelect == 'TSV') {
        codeSnippetText += partial ? partial_build_from_tsv_file_mult_separate : build_from_tsv_file_mult_separate;
    } else if (singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'Yes' && fileTypeSelect == 'CSV') {
        codeSnippetText += partial ? partial_build_from_csv_file_mult_separate : build_from_csv_file_mult_separate;
    }
    return codeSnippetText;
}


function genBuildFromDF(targetFeaturesSeparateDF, singleMultDF, partial) {
    let df_params = '';
    if (targetFeaturesSeparateDF === 'No' && singleMultDF === 'Single DataFrame') {
        df_params = `datasets=<span class=\"highlightText\">df${partial ? '_2': ''}</span>`;
    } else if (targetFeaturesSeparateDF === 'No' && singleMultDF === 'Multiple DataFrames') {
        df_params = `datasets=<span class=\"highlightText\">[df1${partial ? '_2': ''}, df2${partial ? '_2': ''}]</span>`;
    } else if (targetFeaturesSeparateDF === 'Yes' && singleMultDF == 'Single DataFrame') {
        df_params = `datasets=<span class=\"highlightText\">df${partial ? '_2': ''}</span>, target_datasets==<span class=\"highlightText\">y_df${partial ? '_2': ''}</span>`;
    } else if (targetFeaturesSeparateDF === 'Yes' && singleMultDF === 'Multiple DataFrames') {
        df_params = `datasets=<span class=\"highlightText\">[df1${partial ? '_2': ''}, df2${partial ? '_2': ''}]</span>, target_datasets=<span class=\"highlightText\">[y_df1${partial ? '_2': ''}, y_df2${partial ? '_2': ''}]</span>`;
    }
    let codeSnippetText = partial ? `<span class=\"commentText\"># Add additional data to the Coreset tree</span>` : '';
    codeSnippetText += `<span class=\"commentText\"># Build the Coreset Tree</span>        
service_obj.${partial ? 'partial_': ''}build_from_df(${df_params})            
`;

    return codeSnippetText;
}

function genBuildFromNP(singleMultNPY, partial) {
    let np_params = '';

    if (!unsupervisedAlg.includes(alg)) {
        if (singleMultNPY === 'Single numpy array') {
            np_params = `X=<span class=\"highlightText\">X${partial ? '_2' : ''}</span>, y=<span class=\"highlightText\">y${partial ? '_2' : ''}</span>`;
        } else if (singleMultNPY === 'Multiple numpy arrays') {
            np_params = `X=<span class=\"highlightText\">[X1${partial ? '_2' : ''}, X2${partial ? '_2' : ''}]</span>, y=<span class=\"highlightText\">[y1${partial ? '_2' : ''}, y2${partial ? '_2' : ''}]</span>`;
        }
    }else{
        if (singleMultNPY === 'Single numpy array') {
            np_params = `X=<span class=\"highlightText\">X${partial ? '_2' : ''}</span>`;
        } else if (singleMultNPY === 'Multiple numpy arrays') {
            np_params = `X=<span class=\"highlightText\">[X1${partial ? '_2' : ''}, X2${partial ? '_2' : ''}]</span>`;
        }
    }
    let codeSnippetText = partial ? `<span class=\"commentText\"># Add additional data to the Coreset tree</span>` : '';

    return codeSnippetText += `
<span class=\"commentText\"># Build the Coreset Tree</span>        
service_obj.${partial ? 'partial_': ''}build(${np_params})            
`;
}

function genBuild(
    form,
    singleMultFilesDirs,
    targetFeaturesSeparate,
    fileTypeSelect,
    targetFeaturesSeparateDF,
    singleMultDF,
    singleMultNPY,
    partial
) {
    let codeSnippetText = '';

    if (form == 'File') {
        codeSnippetText += genBuildFromFile(
            singleMultFilesDirs = singleMultFilesDirs,
            targetFeaturesSeparate = targetFeaturesSeparate,
            fileTypeSelect = fileTypeSelect,
            partial = partial);

    } else if (form == 'DF') {
        codeSnippetText += genBuildFromDF(
            targetFeaturesSeparateDF = targetFeaturesSeparateDF,
            singleMultDF = singleMultDF,
            partial = partial
        );
    } else if (form == 'NP') {
        codeSnippetText += genBuildFromNP(singleMultNPY = singleMultNPY, partial = partial);
    }
    return codeSnippetText;
}

function genCodeText(dsType,
                     useCases,
                     alg,
                     lib,
                     form,
                     fileTypeSelect,
                     targetFeaturesSeparate,
                     singleMultFilesDirs,
                     targetFeaturesSeparateDF,
                     singleMultDF,
                     singleMultNPY)
{
    const coresetTreeServiceClass = algToClass[alg]

    // useCaseProcessedComment - "train and cleaning" etc.
    // useCaseOptimizedForStr -  init optimized_for value
    let useCaseProcessed = useCases.map(el => {
        if (el==='Model training' || el==='Model maintenance' ){
            return 'training';
        }
        if (el==='Data Cleaning'){
            return 'cleaning';
        }
        if (el==='Model tuning'){
            return 'tuning';
        }
        return '';
    });
    useCaseProcessed = useCaseProcessed.flatMap(el => el==='' ? [] : [el]);
    useCaseProcessed = [...new Set(useCaseProcessed)]
    let useCaseProcessedComment = '';
    if (useCaseProcessed.length == 1){
        useCaseProcessedComment = useCaseProcessed[0];
    }else{
        useCaseProcessedComment = useCaseProcessed.slice(0, -1).join(', ') + ' and ' + useCaseProcessed[useCaseProcessed.length - 1]
    }
    let useCaseOptimizedFor = useCaseProcessed.flatMap(el => el=='tuning' ? [] : [el]);
    let useCaseOptimizedForStr = ''
    if (useCaseOptimizedFor.length == 2){
        useCaseOptimizedForStr = '["training", "cleaning"]';
    }else{
        useCaseOptimizedForStr = '"' + useCaseOptimizedFor[0] + '"';
    }
    ///////////////////

    let service_init = ''
    if (form === 'File' || form ==='DF'){
        console.log(alg, ['Logistic Regression', 'Decision trees classification based'].includes(alg));
        if (['Logistic Regression', 'Decision trees classification based'].includes(alg)){
            service_init = `
<span class=\"commentText\"># Tell the tree how data is structured.
# In this example we have one target column, all other columns are features.
# Replace it with your own target feature name.</span>
data_params = {'target': {'name': '<span class=\"highlightText\">Cover_Type</span>'}}
`
        }else if (['Linear Regression', 'Decision trees regression based'].includes(alg)){
            service_init = `
<span class=\"commentText\"># Tell the tree how data is structured.
# In this example we have one target column, all other columns are features.
# Replace it with your own target feature name.</span>
data_params = {'target': {'name': '<span class=\"highlightText\">tip_amount</span>'}}
`
        }
    }
    if (dsType == 'Tabular'){
        n_instances = '290_000';
        n_classes = '7';
    } else if (dsType == 'NLP'){
        n_instances = '100_000';
        n_classes = '';
    }else{
        n_instances = '1_200_000';
        n_classes = '1_000';
    }

    if (lib == 'XGBoost') {
        lib_import = `
from xgboost import XGBClassifier
`;
        lib_param = '\n    model_cls=XGBClassifier,'
    }else if (lib == 'LightGBM') {
        lib_import = `
from lightgbm import LGBMClassifier
`;
        lib_param = '\n    model_cls=LGBMClassifier,'
    }else if (lib == 'CatBoost') {
        lib_import = `
from catboost import CatBoostClassifier
`;
        lib_param = '\n    model_cls=CatBoostClassifier,'
    }else if (lib == 'Scikit-learn' && alg.includes('Decision trees') ) {
        lib_import = `
from sklearn.ensemble import GradientBoostingClassifier
`;
        lib_param = '\n    model_cls=GradientBoostingClassifier,'
    }else{
        lib_import ='';
        lib_param = '';
    }

    service_init += `${(lib_import !=='') ? lib_import:''}    
<span class=\"commentText\"># Initialize the service for ${useCaseProcessedComment} and build the Coreset tree.
# Change the number of instances ‘n_instances’ to match your dataset.
# The Coreset tree uses the local file system to store its data.
# After this step you will have a new directory .dataheroes_cache</span>
service_obj = ${coresetTreeServiceClass}(${(form === 'File' || form ==='DF') ? `\n    data_params=data_params,`: ''}
    optimized_for=${useCaseOptimizedForStr},
    n_instances=<span class=\"highlightText\">${n_instances}</span>,${(lib_import !=='') ? lib_param:''}
)`

    cleaning_processing=`
<span class=\"commentText\"># Define the classes of interest and the number of samples you want to examine per class (adjust according to your needs).
# Alternatively just pass size=100 (or any other number) to get the top importance samples across all classes.</span>
samples_per_class = <span class=\"highlightText\">{0: 100, 3: 20}</span>
result = service_obj.get_important_samples(class_size=samples_per_class)
important_samples, importance = result['idx'], result["importance"]

<span class=\"commentText\"># Examine the returned samples for mislabels or other anomalies.
# Use update_targets to relabel samples.
# We simulated this by calling a fix_labels function. Replace it with your own method to select the relabeled samples).</span>
indices_relabeled, y_train_relabeled = <span class=\"highlightText\">fix_labels(...)</span>
service_obj.update_targets(indices=indices_relabeled, y=y_train_relabeled)

<span class=\"commentText\"># Use remove_samples to remove samples from the dataset. 
# Replace indices_to_remove with the indices you wish to remove.</span>
service_obj.remove_samples(important_samples[<span class=\"highlightText\">indices_to_remove</span>])
`
let what_adjusting = '';
let metric = '';
let corset_score_expression = '';
let model_params = '';
let tuning_processing = '';

if (alg == 'Logistic Regression'){
    what_adjusting = 'max_iter or using multi_class';
    metric = 'roc_auc_score';
    corset_score_expression = `<span class=\"highlightText\">roc_auc_score</span>(y_test, service_obj.predict_proba(X_test), <span class=\"highlightText\">multi_class=\'ovr\'</span>)`;
    model_params = 'max_iter=200';

    tuning_processing = `<span class=\"commentText\"># To hyperparameter tune your model, use the library’s built-in grid_search function, 
# which would run dramatically faster than GridSearchCV on the entire dataset.
# Adjust the hyperparameters and scoring function to your needs.</span>
param_grid = {
<span class=\"highlightText\">   'penalty' : ['l1','l2'],
   'C'       : np.logspace(-3,3,7),
   'solver'  : ['newton-cg', 'lbfgs', 'liblinear']</span>      
}
scoring = make_scorer(<span</span> class=\"highlightText\">roc_auc_score</span>)

optimal_hyperparameters, trained_model = service_obj.grid_search(
    param_grid=param_grid, 
    scoring=scoring, 
    verbose=2)
`

} else if (alg == 'Linear Regression'){
    what_adjusting = 'n_jobs';
    metric = 'mean_squared_error';
    corset_score_expression = `<span class=\"highlightText\">mean_squared_error</span>(y_test, service_obj.predict_proba(X_test))`;
    model_params = 'n_jobs=2';

    tuning_processing = `<span class=\"commentText\"># To hyperparameter tune your model, use the library’s built-in grid_search function, 
# which would run dramatically faster than GridSearchCV on the entire dataset.
# Adjust the hyperparameters and scoring function to your needs.</span>
param_grid = {
<span class=\"highlightText\">   'fit_intercept': [True, False], 
   'positive': [True, False]</span>      
}
scoring = make_scorer(<span class=\"highlightText\">mean_squared_error</span>)

optimal_hyperparameters, trained_model = service_obj.grid_search(
    param_grid=param_grid, 
    scoring=scoring, 
    verbose=2)
`

} else if (alg == 'K-Means'){
    what_adjusting = 'n_clusters';
    metric = 'silhouette_score';
    corset_score_expression = `<span class=\"highlightText\">silhouette_score</span>(y_test, service_obj.predict(X_test))`;
    model_params = 'n_clusters=10';

    tuning_processing = ''

}else if (alg == 'PCA'){
    what_adjusting = 'n_components';
    metric = '';
    corset_score_expression = `<span class=\"highlightText\">np.sum(coreset_model.explained_variance_ratio_)</span>`;
    model_params = 'n_components=10';

    tuning_processing = ''

}else if (alg == 'SVD'){
    what_adjusting = 'n_iter';
    metric = '';
    corset_score_expression = `<span class=\"highlightText\">np.sum(coreset_model.explained_variance_)</span>`;
    model_params = 'n_iter=8';

    tuning_processing = ''

}else if (alg.includes('Decision')){
    what_adjusting = 'n_estimators';
    metric = 'balanced_accuracy_score';
    corset_score_expression = '<span class=\"highlightText\">balanced_accuracy_score</span>(y_test, service_obj.predict(X_test))';
    model_params = 'n_estimators=500';
    tuning_processing = `<span class=\"commentText\"># To hyperparameter tune your model, use the library’s built-in grid_search function, 
# which would run dramatically faster than GridSearchCV on the entire dataset.
# Adjust the hyperparameters and scoring function to your needs.</span>
param_grid = {
<span class=\"highlightText\">   'learning_rate': [0.1, 0.01],
   'n_estimators': [250, 500, 1000],
   'max_depth': [4, 6]</span>  
}
scoring = make_scorer(<span class=\"highlightText\">balanced_accuracy_score</span>)

optimal_hyperparameters, trained_model = service_obj.grid_search(
    param_grid=param_grid, 
    scoring=scoring, 
    verbose=2
    )
`
}

let training_processing=`<span class=\"commentText\"># fit a ${alg} model using ${lib} directly on the Coreset tree.
# Try a few levels of the Coreset tree to find the optimal one.
# Provide the same parameters to the fit, predict and predict_proba Coreset methods  
# as you would provide ${lib} (adjusting ${what_adjusting}).</span>${metric !== '' ? `\nfrom sklearn.metrics import <span class=\"highlightText\">${metric}</span>` : ''}
for tree_level in range(3):
   coreset_model = service_obj.fit(level=tree_level, <span class=\"highlightText\">${model_params}</span>)
   coreset_score = ${corset_score_expression}
`

    let codeSnippetText = '';
    if (dsType != 'Tabular' && dsType != 'NLP'){
        codeSnippetText += image_dataset_comment_1;
    }
    codeSnippetText += `from dataheroes import ${coresetTreeServiceClass} \n`;

    codeSnippetText += service_init;

    codeSnippetText += '\n'+ genBuild(
        form=form,
        singleMultFilesDirs=singleMultFilesDirs,
        targetFeaturesSeparate=targetFeaturesSeparate,
        fileTypeSelect=fileTypeSelect,
        //DF
        targetFeaturesSeparateDF=targetFeaturesSeparateDF,
        singleMultDF=singleMultDF,
        //NP
        singleMultNPY=singleMultNPY,
        partial=false
    );

    if (useCases.includes('Data Cleaning')) {
        codeSnippetText += cleaning_processing;
    }

    if (useCases.includes('Model maintenance')) {
        codeSnippetText += '\n'+ genBuild(
        form=form,
        singleMultFilesDirs=singleMultFilesDirs,
        targetFeaturesSeparate=targetFeaturesSeparate,
        fileTypeSelect=fileTypeSelect,
        //DF
        targetFeaturesSeparateDF=targetFeaturesSeparateDF,
        singleMultDF=singleMultDF,
        //NP
        singleMultNPY=singleMultNPY,
        partial=true
    );
    }

    if (useCases.includes('Model training')) {
        codeSnippetText += '\n'+training_processing;
    }

    if (useCases.includes('Model tuning')) {
        codeSnippetText += '\n'+tuning_processing;
    }




    let finalComment = ''
    if (alg == 'Linear Regression'){
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a linear regression model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_linear_regression_tabular_data_yellowtaxi.ipynb\">link</a>.</span>  
`
    } else if (alg == 'Logistic Regression'){
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a logistic regression model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_logistic_regression_tabular_data_covertype.ipynb\">link</a>.</span>  
`
    } else if (alg == 'PCA'){
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a PCA model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_pca_tabular_data_higgs.ipynb\">link</a>.</span>  
`
    } else if (alg == 'K-Means'){
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a K_Means model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_kmeans_tabular_data_covertype.ipynb\">link</a>.</span>  
`
    } else if (alg == 'SVD'){
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a logistic regression model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_logistic_regression_tabular_data_covertype.ipynb\">link</a>.</span>  
`
    } else if (lib == 'LightGBM'){
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a LightGBM model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_lightGBM_tabular_data_pokerhand.ipynb\">link</a>.</span>  
`
    }else if (alg.includes('Decision')) {
        finalComment = `<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree and train a XGBoost model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_xgboost_tabular_data_pokerhand.ipynb\">link</a>.</span>  
`
    }

    codeSnippetText += '\n'+finalComment;

    if (useCases.includes('Data Cleaning')){
        codeSnippetText += '\n'+`<span class=\"commentText\"># For a full notebook showing how to build a Coreset tree for cleaning purposes for the ImageNet dataset, visit this <a target="_blank" href="https://github.com/Data-Heroes/dataheroes/blob/master/examples/cleaning/data_cleaning_image_classification_imagenet.ipynb">link</a>.

# For a full notebook showing how to build a Coreset tree for cleaning purposes comparing cleaning using the DataHeroes 
# library to random cleaning for the CIFAR10 dataset, visit this <a target="_blank" href=https://github.com/Data-Heroes/dataheroes/blob/master/examples/cleaning/data_cleaning_coreset_vs_random_image_classification_cifar10.ipynb">link</a>.</span>    
`
    }

    return codeSnippetText;
}