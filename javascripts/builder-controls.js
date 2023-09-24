const dtypeSelect = document.getElementById('Dataset type');
const mlAlgSelect = document.getElementById('ML algorithm');
const trainLibSelect = document.getElementById('Library used to train models');
const useCaseCheckList = document.getElementById('Use Case Check');

//const tabularUseCases = ["Model training", "Model tuning", "Model maintenance", "Data cleaning"];
const tabularUseCases = ["Model training and tuning", "Model maintenance", "Data cleaning"];
const nonTabularUseCases = ["Data cleaning"];

const tabularMLAlg = ["Linear Regression", "Logistic Regression", "K-Means", "PCA", "SVD", "Decision trees classification based", "Decision trees regression based"].sort();
const nonTabularMLAlg = ["Deep learning classification", "Deep learning regression"].sort();

const tabularTrainLib = ["XGBoost", "LightGBM", "CatBoost", "Scikit-learn"];
const nonTabularTrainLib = ["PyTorch", "TensorFlow (currently used for feature extraction)"];

function getSelectOptions(elements){
    let result = '';
    for (el of elements){
        if (elements[0] === el) {
            result += '<option value="' + el + '">' + el + '</option>';
        }else{
            result += '<option value="' + el + '">' + el + '</option>';
        }
    }
    return result;
}

function getUseCasesList(){
    let useCasesList = [];
    for (let v in tabularUseCases) {
        el = document.getElementById(tabularUseCases[v]);
        if (el && el.checked) {
            useCasesList.push(tabularUseCases[v]);
        }
    }
    return useCasesList;
}

function getSelectOptionsCheckbox(elements){

    let result = '';
    for (el of elements){
        let checked = '';
        if (result === ''){
            checked = 'checked';
        }
        let tooltip = '';
        if (el =='Model training and tuning'){
            tooltip = 'Training and hyperparameter tuning models';
        } else if (el=='Model maintenance'){
            tooltip = 'Training and hyperparameter tuning models';
        } else if (el=='Data cleaning'){
            tooltip = 'Identifying mislabels and outliers';
        }
        result += `<div class="checkbox-full"><input ${checked} type="checkbox" onclick="handleUseCaseChange(this);" name="${el}" id="${el}"><label for="${el}">${el}
<span data-toggle='tooltip' class='checkbox-tooltip' title='${tooltip}'>
<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M478-240q21 0 35.5-14.5T528-290q0-21-14.5-35.5T478-340q-21 0-35.5 14.5T428-290q0 21 14.5 35.5T478-240Zm-36-154h74q0-33 7.5-52t42.5-52q26-26 41-49.5t15-56.5q0-56-41-86t-97-30q-57 0-92.5 30T342-618l66 26q5-18 22.5-39t53.5-21q32 0 48 17.5t16 38.5q0 20-12 37.5T506-526q-44 39-54 59t-10 73Zm38 314q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"/></svg>
</span> 

</label>






</div>`;

    }

    return result;
}

function generateText(){
    const warningText = document.getElementById('warning_text');
    const warningDiv = document.getElementById('warningDiv');
    const codeSnippetDiv = document.getElementById('codeSnippet');

    warningDiv.style.display = 'none';
    codeSnippetDiv.style.display = 'none';

    function isAllFilled(){
        if (document.getElementById('datasetFormSelect').value==='File'){
            if (document.getElementById('File Type').value==='' ||
                document.getElementById('targetFeaturesSeparate').value==='' ||
                document.getElementById('singleMultFilesDirs').value===''
                ){
                return false;
            }

        }else if (document.getElementById('datasetFormSelect').value==='DF'){
            if (document.getElementById('targetFeaturesSeparateDF').value==='' ||
                document.getElementById('singleMultDF').value===''
                ){
                return false;
            }
        }else if (document.getElementById('datasetFormSelect').value==='NP'){
            if (document.getElementById('singleMultNPY').value===''
                ){
                return false;
            }
        }
        return true;
    }
    let useCasesList = getUseCasesList();
    if (dtypeSelect.value === ''){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Select a dataset type";
    }else if (useCasesList.length === 0){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Please select at least a single Use case";
    }else if (mlAlgSelect.value === ''){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Select a ML algorithm";
    }else if (trainLibSelect.value === ''){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Select a library used to train models";
    }else if (document.getElementById('datasetFormSelect').value === ''){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Select a form of dataset(s)";
    }else if (mlAlgSelect.value === ''){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Select a ML algorithm";
    }else if (!isAllFilled() ){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Answer all questions";
    }else{
        //success!!!
        codeSnippetDiv.style.display = 'block';
        let useCasesList = getUseCasesList();

        datasetFormSelect = document.getElementById('datasetFormSelect')
        fileTypeSelect = document.getElementById('File Type')
        targetFeaturesSeparate = document.getElementById('targetFeaturesSeparate')
        singleMultFilesDirs = document.getElementById('singleMultFilesDirs')
        targetFeaturesSeparateDF = document.getElementById('targetFeaturesSeparateDF')
        singleMultDF = document.getElementById('singleMultDF')
        singleMultNPY = document.getElementById('singleMultNPY')


        document.getElementById('codeSnippetText').innerHTML = genCodeText(
                                                                dsType=dtypeSelect.value,
                                                                useCases=useCasesList,
                                                                alg=mlAlgSelect.value,
                                                                lib=trainLibSelect.value,
                                                                form=datasetFormSelect.value,

                                                                fileTypeSelect=fileTypeSelect.value,
                                                                targetFeaturesSeparate=targetFeaturesSeparate.value,
                                                                singleMultFilesDirs=singleMultFilesDirs.value,

                                                                targetFeaturesSeparateDF=targetFeaturesSeparateDF.value,
                                                                singleMultDF=singleMultDF.value,

                                                                singleMultNPY=singleMultNPY.value
                                                            );

    }

}


function handleUseCaseChange(){
    if (document.getElementById('Model maintenance').checked
        &&
        !document.getElementById('Model training and tuning').checked
    ){
        document.getElementById('Model training and tuning').click();
    }
    document.getElementById('Model training and tuning').disabled =
        document.getElementById('Model maintenance').checked;

    generateText();
}

const selects = document.querySelectorAll('select');
selects.forEach(select => {
    select.addEventListener('change', function() {
        generateText();
    });
});


function handleDType(){
    const selectedDtype = dtypeSelect.value;
    document.getElementById('tabularOptions').style.display = 'none';
    if (selectedDtype === 'Tabular'){
        document.getElementById('tabularOptions').style.display = 'block';
        useCaseCheckList.innerHTML = getSelectOptionsCheckbox(tabularUseCases);
        mlAlgSelect.innerHTML = getSelectOptions(tabularMLAlg);
        trainLibSelect.innerHTML = getSelectOptions(tabularTrainLib);
    }else{
        //defaults for non-tabular dataset
        document.getElementById('File Type').value = 'NPY';
        document.getElementById('targetFeaturesSeparate').value = 'Yes';

        useCaseCheckList.innerHTML = getSelectOptionsCheckbox(nonTabularUseCases);
        mlAlgSelect.innerHTML = getSelectOptions(nonTabularMLAlg);
        trainLibSelect.innerHTML = getSelectOptions(nonTabularTrainLib);
    }
}

function handleAlg(){
    const selectedDtype = dtypeSelect.value;
    const selectedAlg = mlAlgSelect.value;
    // Clear previous options
    if (selectedDtype === 'Tabular' && !selectedAlg.includes('Decision trees')){
        trainLibSelect.innerHTML = getSelectOptions(['Scikit-learn']);
        trainLibSelect.value = 'Scikit-learn';
    }else if (selectedDtype === 'Tabular' && selectedAlg.includes('Decision trees')){
        trainLibSelect.innerHTML = getSelectOptions(tabularTrainLib);
    }else{
        trainLibSelect.innerHTML = getSelectOptions(nonTabularTrainLib);
    }
    ////targetFeaturesSeparateDF targetFeaturesSeparate
    if (['K-Means', 'PCA', 'SVD'].includes(selectedAlg)){
        document.getElementById('targetFeaturesSeparate').innerHTML = getSelectOptions(['No']);
        document.getElementById('targetFeaturesSeparateDF').innerHTML = getSelectOptions(['No']);
    }else{
        document.getElementById('targetFeaturesSeparate').innerHTML = getSelectOptions(['No', 'Yes']);
        document.getElementById('targetFeaturesSeparateDF').innerHTML = getSelectOptions(['No', 'Yes']);

        if (dtypeSelect.value !=='Tabular'){
            //default value for non-tabular
            document.getElementById('targetFeaturesSeparate').value = 'Yes';
        }
    }
    generateText();
}

dtypeSelect.addEventListener('change', function() {
    handleDType();
    handleAlg();
});

mlAlgSelect.addEventListener('change', function() {
    handleAlg();
});

function handleForm(){
    const selectedDatasetForm = datasetFormSelect.value;
    optionFileContent.style.display = 'none';
    optionDFContent.style.display = 'none';
    optionNPContent.style.display = 'none';

    if (selectedDatasetForm === 'File'){
        optionFileContent.style.display = 'block';
    }else if (selectedDatasetForm ==='DF') {
        optionDFContent.style.display = 'block';
    }else if (selectedDatasetForm ==='NP') {
        optionNPContent.style.display = 'block';
    }


}






datasetFormSelect.addEventListener('change', function() {
    handleForm();
});

handleDType();
handleAlg();
handleForm();
generateText();


