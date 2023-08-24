const dtypeSelect = document.getElementById('Dataset type');
const mlAlgSelect = document.getElementById('ML algorithm');
const trainLibSelect = document.getElementById('Library used to train models');
const useCaseCheckList = document.getElementById('Use Case Check');

const tabularUseCases = ["Model training", "Model tuning", "Model maintenance", "Data cleaning"];
const nonTabularUseCases = ["Data cleaning"];

const tabularMLAlg = ["Linear Regression", "Logistic Regression", "K-Means", "PCA", "SVD", "Decision trees classification based", "Decision trees regression based"].sort();
const tabularTuningAlg = ["Linear Regression", "Logistic Regression", "Decision trees classification based", "Decision trees regression based"].sort();
const nonTabularMLAlg = ["Deep learning classification", "Deep learning regression"].sort();

const tabularTrainLib = ["XGBoost", "LightGBM", "CatBoost", "Scikit-learn"];
const nonTabularTrainLib = ["PyTorch", "TensorFlow (currently used for feature extraction)"];

function getSelectOptions(elements){
    let result = '';
    for (el of elements){
        if (elements[0] == el) {
            result += '<option value="' + el + '">' + el + '</option>';
        }else{
            result += '<option value="' + el + '">' + el + '</option>';
        }
    }
    return result;
}

function getUseCasesList(){
    var useCasesList = [];
    for (v in tabularUseCases) {
        el = document.getElementById(tabularUseCases[v]);
        if (el && el.checked) {
            useCasesList.push(tabularUseCases[v]);
        }
    }
    return useCasesList;
}

function getSelectOptionsCheckbox(elements){

    result = '';
    for (el of elements){
        checked = '';
        if (result === ''){
            checked = 'checked';
        }
        result += `<div class="checkbox-full"><input ${checked} type="checkbox" onclick="handleUseCaseChange(this);" name="${el}" id="${el}"><label for="${el}">${el}</label></div>`;

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
    useCasesList = getUseCasesList();
    if (dtypeSelect.value === ''){
        warningDiv.style.display = 'block';
        warningText.innerHTML = "Select a dataset type";
    }else if (useCasesList.length == 0){
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
        useCases = getUseCasesList();
        useCaseStr = useCases.join(', ');



        datasetFormSelect = document.getElementById('datasetFormSelect')
        fileTypeSelect = document.getElementById('File Type')
        targetFeaturesSeparate = document.getElementById('targetFeaturesSeparate')
        singleMultFilesDirs = document.getElementById('singleMultFilesDirs')
        targetFeaturesSeparateDF = document.getElementById('targetFeaturesSeparateDF')
        singleMultDF = document.getElementById('singleMultDF')
        singleMultNPY = document.getElementById('singleMultNPY')


        document.getElementById('codeSnippetText').innerHTML = genCodeText(
                                                                dsType=dtypeSelect.value,
                                                                useCases=useCases,
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


function handleUseCases(){
    const selectedDtype = dtypeSelect.value;
    if (selectedDtype == 'Tabular'){
        if (!document.getElementById('Model tuning').checked) {
            mlAlgSelect.innerHTML = getSelectOptions(tabularMLAlg);
        }else{
            mlAlgSelect.innerHTML = getSelectOptions(tabularTuningAlg);
        }
    }else{
        mlAlgSelect.innerHTML = getSelectOptions(nonTabularMLAlg);
    }
}


function handleUseCaseChange(cb){
    //handleUseCases();
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
    // Clear previous options
    if (selectedDtype == 'Tabular'){
        useCaseCheckList.innerHTML = getSelectOptionsCheckbox(tabularUseCases);
        mlAlgSelect.innerHTML = getSelectOptions(tabularMLAlg);
        trainLibSelect.innerHTML = getSelectOptions(tabularTrainLib);
    }else if (selectedDtype==='') {
        useCaseCheckList.innerHTML = '                           ';
        mlAlgSelect.innerHTML = '                               ';
    }else{
        useCaseCheckList.innerHTML = getSelectOptionsCheckbox(nonTabularUseCases);
        mlAlgSelect.innerHTML = getSelectOptions(nonTabularMLAlg);
        trainLibSelect.innerHTML = getSelectOptions(nonTabularTrainLib);
    }
}

function handleAlg(){
    const selectedDtype = dtypeSelect.value;
    const selectedAlg = mlAlgSelect.value;
    // Clear previous options
    if (selectedDtype == 'Tabular' && !selectedAlg.includes('Decision trees')){
        trainLibSelect.innerHTML = getSelectOptions(['Scikit-learn']);
        trainLibSelect.value = 'Scikit-learn';
    }else if (selectedDtype == 'Tabular' && selectedAlg.includes('Decision trees')){
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


