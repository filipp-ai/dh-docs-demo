image_dataset_comment_1 = `
# A prerequisite to use this Coreset is to extract the features embeddings from your images.
# In order to do so, drop the last classification layer from your pre-trained network, so the output would be the embeddings instead of the class distribution.
# To see how we extracted the feature embeddings from the ImageNet dataset using ResNet18 and PyTorch, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/feature_extraction_scripts/feature_extraction_imagenet1k_resnet18_pytorch.py\">link</a>.
# To see how we extracted the feature embeddings from the CIFAR10 dataset using ResNet18 and TensorFlow /Keras, visit this <a target=\"_blank\" href=https://github.com/Data-Heroes/dataheroes/blob/master/examples/feature_extraction_scripts/feature_extraction_cifar10_resnet18_tensorflow_keras.py">link</a>.

`

const algToClass = {
        "Linear Regression": 'CoresetTreeServiceLR',
        "Logistic Regression": 'CoresetTreeServiceLG',
        "K-Means": 'CoresetTreeServiceKMeans',
        "PCA": 'CoresetTreeServicePCA',
        "SVD": 'CoresetTreeServiceSVD',
        "Decision trees classification based": 'CoresetTreeServiceDTC',
        "Decision trees regression based": 'CoresetTreeServiceDTC',
        "Deep learning classification": 'CoresetTreeServiceLG',
        "Deep learning regression": 'CoresetTreeServiceLR',
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

service_training_init = `
data_params = {'target': {'name': '<span class=\"highlightText\">Cover_Type</span>'}}

# Initialize the service for training and tuning and build the Coreset tree.
# Change the number of instances ‘n_instances’ and number of classes ‘n_classes’ to match your dataset.
# The Coreset tree uses the local file system to store its data.
# After this step you will have a new directory .dataheroes_cache
service_obj = ${coresetTreeServiceClass}(data_params=data_params, 
                                   optimized_for="training", 
                                   save_all=True,
                                   n_instances=<span class=\"highlightText\">290_000</span>, 
                                   n_classes=<span class=\"highlightText\">7</span>)
    `

service_cleaning_init = `
# Initialize the service for cleaning and build the Coreset tree.
# Change the number of instances ‘n_instances’ and number of classes ‘n_classes’ to match your dataset.
# The Coreset tree uses the local file system to store its data.
# After this step you will have a new directory .dataheroes_cache
service_obj = ${coresetTreeServiceClass}(optimized_for="cleaning",
                                   n_instances=<span class=\"highlightText\">1_200_000</span>, 
                                   n_classes=<span class=\"highlightText\">1000</span>)
    `

let build_from_npy_file_single_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.
x_train_file = <span class=\"highlightText\">data_generated / 'x_train.npy'</span>
y_train_file = <span class=\"highlightText\">data_generated / 'y_train.npy'</span>

service_obj.build_from_file(x_train_file, y_train_file, reader_f=read_npy)

`

let build_from_csv_file_single_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
x_train_file = <span class=\"highlightText\">data_generated / 'x_train.csv'</span>
y_train_file = <span class=\"highlightText\">data_generated / 'y_train.csv'</span>

service_obj.build_from_file(x_train_file, y_train_file)

`

let build_from_tsv_file_single_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
x_train_file = <span class=\"highlightText\">data_generated / 'x_train.tsv'</span>
y_train_file = <span class=\"highlightText\">data_generated / 'y_train.tsv'</span>

service_obj.build_from_file(x_train_file, y_train_file, sep='\\t')

`

let build_from_npy_file_single = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.
data_file = <span class=\"highlightText\">data_generated / 'data.npy'</span>

service_obj.build_from_file(data_file, reader_f=read_npy)

`

let build_from_csv_file_single = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
data_file = <span class=\"highlightText\">data_generated / 'data.csv'</span>

service_obj.build_from_file(data_file)

`

let build_from_tsv_file_single = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
data_file = <span class=\"highlightText\">data_generated / 'data.tsv'</span>

service_obj.build_from_file(data_file, sep='\\t')

`

let build_from_npy_file_mult = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.
data_file1 = <span class=\"highlightText\">data_generated / 'data1.npy'</span>
data_file2 = <span class=\"highlightText\">data_generated / 'data2.npy'</span>

service_obj.build_from_file([data_file1, data_file2], reader_f=read_npy)

`

let build_from_csv_file_mult = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
data_file1 = <span class=\"highlightText\">data_generated / 'data1.npy'</span>
data_file2 = <span class=\"highlightText\">data_generated / 'data2.npy'</span>

service_obj.build_from_file([data_file1, data_file2])

`

let build_from_tsv_file_mult = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
data_file1 = <span class=\"highlightText\">data_generated / 'data1.npy'</span>
data_file2 = <span class=\"highlightText\">data_generated / 'data2.npy'</span>

service_obj.build_from_file([data_file1, data_file2], sep='\\t')

`

let build_from_npy_file_mult_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
# and build the Coreset tree using the read_npy reader.
x_train_file1 = <span class=\"highlightText\">data_generated / 'x_train1.npy'</span>
x_train_file2 = <span class=\"highlightText\">data_generated / 'x_train2.npy'</span>
y_train_file1 = <span class=\"highlightText\">data_generated / 'y_train1.npy'</span>
y_train_file2 = <span class=\"highlightText\">data_generated / 'y_train2.npy'</span>

service_obj.build_from_file([x_train_file1, x_train_file2], [y_train_file1, y_train_file2], reader_f=read_npy)

`

let build_from_csv_file_mult_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
x_train_file1 = <span class=\"highlightText\">data_generated / 'x_train1.csv'</span>
x_train_file2 = <span class=\"highlightText\">data_generated / 'x_train2.csv'</span>
y_train_file1 = <span class=\"highlightText\">data_generated / 'y_train1.csv'</span>
y_train_file2 = <span class=\"highlightText\">data_generated / 'y_train2.csv'</span>

service_obj.build_from_file([x_train_file1, x_train_file2], [y_train_file1, y_train_file2])

`

let build_from_tsv_file_mult_separate = `
# Prepare the definitions of path variables for the dataset files (replace with your directory name and file names) 
x_train_file1 = <span class=\"highlightText\">data_generated / 'x_train1.tsv'</span>
x_train_file2 = <span class=\"highlightText\">data_generated / 'x_train2.tsv'</span>
y_train_file1 = <span class=\"highlightText\">data_generated / 'y_train1.tsv'</span>
y_train_file2 = <span class=\"highlightText\">data_generated / 'y_train2.tsv'</span>

service_obj.build_from_file([x_train_file1, x_train_file2], [y_train_file1, y_train_file2], sep='\\t')

`


build_from_dir_single = `
# Prepare the data directory (replace with your directory name) and build the Coreset tree from the files in it.
from pathlib import Path
data_dir = Path("<span class=\"highlightText\">data_dir</span>")
service_obj.build_from_file(data_dir)
`

build_from_dir_single_separate = `
# Prepare the data and target directories (replace with your directories names) and build the Coreset tree from the files in it.
from pathlib import Path
data_dir = Path("<span class=\"highlightText\">data_dir</span>")
target_dir = Path("<span class=\"highlightText\">target_dir</span>")
service_obj.build_from_file(file_path=data_dir, target_file_path=target_dir)
`

build_from_dir_mult = `
# Prepare the data and target directories (replace with your directories names) and build the Coreset tree from the files in it.
from pathlib import Path
data_dir1 = Path("<span class=\"highlightText\">data_dir1</span>")
data_dir2 = Path("<span class=\"highlightText\">data_dir2</span>")
service_obj.build_from_file(file_path=[data_dir1, data_dir2]])
`

build_from_dir_mult_separate = `
# Prepare the data and target directories (replace with your directories names) and build the Coreset tree from the files in it.
from pathlib import Path
data_dir1 = Path("<span class=\"highlightText\">data_dir1</span>")
data_dir2 = Path("<span class=\"highlightText\">data_dir2</span>")
target_dir1 = Path("<span class=\"highlightText\">target_dir1</span>")
target_dir2 = Path("<span class=\"highlightText\">target_dir2</span>")
service_obj.build_from_file(file_path=[data_dir1, data_dir2], target_file_path=[target_dir1, target_dir2])
`

cleaning_processing=`
# Define the classes of interest and the number of samples you want to examine per class (adjust according to your needs).
# Alternatively just pass size=100 (or any other number) to get the top importance samples across all classes.
samples_per_class=<span class=\"highlightText\">{3: 33, 4: 33, 5: 33}</span>
result = service_obj.get_important_samples(class_size=samples_per_class)
important_samples, importance = result['idx'], result["importance"]

# Examine the returned samples for mislabels or other anomalies.
# Use update_targets to relabel samples.
# We simulated this by calling a fix_labels function. Replace it with your own method to select the relabeled samples).
<span class=\"highlightText\">indices_relabeled, y_train_relabeled = fix_labels(...)</span>
service_obj.update_targets(indices=<span class=\"highlightText\">indices_relabeled</span>, y=<span class=\"highlightText\">y_train_relabeled</span>)

# Use remove_samples to remove samples from the dataset. 
# Replace indices_to_remove with the indices you wish to remove.
service_obj.remove_samples(important_samples[<span class=\"highlightText\">indices_to_remove</span>])

# For a full notebook showing how to build a Coreset tree for cleaning purposes for the ImageNet dataset, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/cleaning/data_cleaning_image_classification_imagenet.ipynb">link</a>.
# For a full notebook showing how to build a Coreset tree for cleaning purposes comparing cleaning using the DataHeroes 
# library to random cleaning for the CIFAR10 dataset, visit this <a target=\"_blank\" href=https://github.com/Data-Heroes/dataheroes/blob/master/examples/cleaning/data_cleaning_coreset_vs_random_image_classification_cifar10.ipynb">link</a>.    
`

//const nonTabularTrainLib = ["PyTorch", "TensorFlow (currently used for feature extraction)"];
let set_model_cls = '';
if (lib === 'Scikit-learn'){
set_model_cls = `from sklearn.linear_model import LogisticRegression # <span style="background-color: #ef5552">why we need this import?</span>
`
}else if (lib === 'XGBoost'){
set_model_cls = `from xgboost import XGBClassifier
service_obj.model_cls = XGBClassifier
`
}else if (lib === 'LightGBM'){
set_model_cls = `from lightgbm import LGBMClassifier
service_obj.model_cls = LGBMClassifier
`
}else if (lib === 'CatBoost'){
set_model_cls = `from catboost import CatBoostClassifier
service_obj.model_cls = CatBoostClassifier
`
}

training_processing=`
# Train a ${alg} model using ${lib}.
${set_model_cls}
# fit a ${alg} model using ${lib} directly on the Coreset tree. 
# Provide the same parameters to the fit method as you would provide ${lib} (adjusting max_iter in the example).
# Select the Coreset tree level you wish to fit on.
coreset_model = service_obj.fit(level=0, <span class=\"highlightText\">max_iter=200</span>)

# To hyperparameter tune your model, use the library’s built-in grid_search function, which would run dramatically 
# faster than GridSearchCV on the entire dataset.
# Adjust the hyperparameters and scoring function to your needs.
param_grid = {
<span class=\"highlightText\">   'penalty' : ['l1','l2'],
   'C'       : np.logspace(-3,3,7),
   'solver'  : ['newton-cg', 'lbfgs', 'liblinear']</span>   
}
<span class=\"highlightText\">balanced_accuracy_scoring = make_scorer(balanced_accuracy_score)</span>

optimal_hyperparameters, trained_model = service_obj.grid_search(param_grid=param_grid, scoring=balanced_accuracy_scoring, verbose=2)

# For a full notebook showing how to build a Coreset tree and train a logistic regression model on it, visit this <a target=\"_blank\" href=\"https://github.com/Data-Heroes/dataheroes/blob/master/examples/build_and_train/build_and_train_logistic_regression_tabular_data_covertype.ipynb\">link</a>.  
`

    let codeSnippetText = '';
    if (dsType != 'Tabular' && dsType != 'NLP'){
        codeSnippetText += image_dataset_comment_1;
    }
    codeSnippetText += `from dataheroes import ${coresetTreeServiceClass} \n`;

    if (useCases[0] != 'Data Cleaning'){
        codeSnippetText += service_training_init;
    }else{
        codeSnippetText += service_cleaning_init;
    }

    if (form=='File'){
        if(singleMultFilesDirs==='Single Directory' && targetFeaturesSeparate==='No'){
        codeSnippetText += build_from_dir_single
        }else if(singleMultFilesDirs==='Single Directory' && targetFeaturesSeparate==='Yes'){
            codeSnippetText += build_from_dir_single_separate

        }else if(singleMultFilesDirs==='Multiple Directories' && targetFeaturesSeparate==='No'){
            codeSnippetText += build_from_dir_mult
        }else if(singleMultFilesDirs==='Multiple Directories' && targetFeaturesSeparate==='Yes'){
            codeSnippetText += build_from_dir_mult_separate

        }else if(singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'Yes' && fileTypeSelect=='NPY'){
            codeSnippetText += build_from_npy_file_single_separate;
        }else if(singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'Yes' && fileTypeSelect=='TSV'){
            codeSnippetText += build_from_tsv_file_single_separate;
        }else if(singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'Yes' && fileTypeSelect=='CSV'){
            codeSnippetText += build_from_csv_file_single_separate;

        }else if(singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'No' && fileTypeSelect=='NPY'){
            codeSnippetText += build_from_npy_file_single;
        }else if(singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'No' && fileTypeSelect=='TSV'){
            codeSnippetText += build_from_tsv_file_single;
        }else if(singleMultFilesDirs === 'Single File' && targetFeaturesSeparate === 'No' && fileTypeSelect=='CSV'){
            codeSnippetText += build_from_csv_file_single;

        }else if(singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'No' && fileTypeSelect=='NPY'){
            codeSnippetText += build_from_npy_file_mult;
        }else if(singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'No' && fileTypeSelect=='TSV'){
            codeSnippetText += build_from_tsv_file_mult;
        }else if(singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'No' && fileTypeSelect=='CSV'){
            codeSnippetText += build_from_csv_file_mult;
        }else if(singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'Yes' && fileTypeSelect=='NPY'){
            codeSnippetText += build_from_npy_file_mult_separate;
        }else if(singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'Yes' && fileTypeSelect=='TSV'){
            codeSnippetText += build_from_tsv_file_mult_separate;
        }else if(singleMultFilesDirs === 'Multiple Files' && targetFeaturesSeparate === 'Yes' && fileTypeSelect=='CSV'){
            codeSnippetText += build_from_csv_file_mult_separate;
        }else{
            codeSnippetText += `
# <span class=\"highlightText\">TODO FILE SECTION</span> 
    `;
        }
    }else if (form=='DF'){
        let df_params='';
        if (targetFeaturesSeparateDF==='No'  && singleMultDF==='Single DataFrame'){
            df_params = 'datasets=<span class=\"highlightText\">df</span>';
        }else if (targetFeaturesSeparateDF==='No'  && singleMultDF==='Multiple DataFrames'){
            df_params = 'datasets=<span class=\"highlightText\">[df1, df2]</span>';
        }else if (targetFeaturesSeparateDF==='Yes'  && singleMultDF=='Single DataFrame'){
            df_params = 'datasets=<span class=\"highlightText\">df</span>, target_datasets==<span class=\"highlightText\">y_df</span>';
        }else if (targetFeaturesSeparateDF==='Yes'  && singleMultDF==='Multiple DataFrames'){
            df_params = 'datasets=<span class=\"highlightText\">[df1, df2]</span>, target_datasets=<span class=\"highlightText\">[y_df1, y_df2]</span>';
        }

        codeSnippetText += `
servise_obj.build_from_df(${df_params})            
`;
    }else if (form=='NP'){
        let np_params='';
        if (singleMultNPY==='Single numpy array'){
            np_params = 'X=<span class=\"highlightText\">X</span>, y=<span class=\"highlightText\">y</span>';
        }else if (singleMultNPY==='Multiple numpy arrays'){
            np_params = 'X=<span class=\"highlightText\">[X1, X2]</span>, y=<span class=\"highlightText\">[y1, y2]</span>';
        }

        codeSnippetText += `
servise_obj.build(${np_params})            
`;
    }


    if (useCases[0] =='Data Cleaning') {
        codeSnippetText += cleaning_processing
    }else{
        //TODO
        codeSnippetText += training_processing
    }
    return codeSnippetText;
}