# kidney_disease_classification
# End_to_End_ml_project_chest_CT_scan

# Workflow
1. Update config.yaml
2. update secrets.yaml [optional]
3. update params.yaml
4. update the entity
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update the main.py
9. update the dvc.yaml


# HOw to run?

### STEP:

### STEP 01- clone the repository


```bash
https://github.com/proshanta000/kidney_disease_classification.git
```

### STEP 02- creat a conda environment after opening the respository

```bash
conda create -p venv python==3.11 -y
```
```bash
conda activate venv/
```
### STEP 03- install the requirements
```bash
pip install -r requirements.txt
```
### Dagshub
[dagshub](https://dagshub.com/)

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/proshanta000/kidney_disease_classification.mlflow

export MLFLOW_TRACKING_USERNAME= proshanta000

export MLFLOW_TRACKING_PASSWORD= ******
```
