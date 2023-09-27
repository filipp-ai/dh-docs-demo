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
            tooltip = 'Adding data and retraining the models';
        } else if (el=='Data cleaning'){
            tooltip = 'Identifying mislabels and outliers';
        }
        result += `<div class="checkbox-full"><input ${checked} type="checkbox" onclick="handleUseCaseChange(this);" name="${el}" id="${el}"><label for="${el}" data-toggle='tooltip' title='${tooltip}'>${el}
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
    // messy restoring of Model_training_and_tuning value!
    const modelTrainingStateName = "Model_training_and_tuning";

    if (!document.getElementById('Model maintenance').checked &&
        !document.getElementById('Model training and tuning').disabled
    ) {
        // 'Model training and tuning' - store its value
        let stateValue = document.getElementById('Model training and tuning').checked ? '1' : '0';
        localStorage.setItem(modelTrainingStateName, stateValue);
    }

    if (!document.getElementById('Model maintenance').checked &&
        document.getElementById('Model training and tuning').disabled
    ) {
        // if we are enabling 'Model training and tuning' - RE-store its value
        let trainingOriginalChecked = localStorage.getItem(modelTrainingStateName) == '1';
        if (trainingOriginalChecked !== document.getElementById('Model training and tuning').checked
        ) {
            document.getElementById('Model training and tuning').disabled = false;
            document.getElementById('Model training and tuning').click();
        }

    }
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


