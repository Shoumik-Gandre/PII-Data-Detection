# Personally Identifiable Information Detection from Educational Data

This project detects personally identifiable information (PII) from student essays using named entity recognition.

PII data detection offers several important benefits. First, it helps organizations locate sensitive personal data like social security numbers, addresses, dates of birth, and financial information within their systems. Detecting this data allows organizations to better secure it and reduce the risks from data breaches or unauthorized access. Second, PII detection supports data privacy compliance. Laws and regulations generally require protection of personal information and notification if such data is improperly accessed. By finding all PII data within corporate systems, organizations can better monitor access to it and demonstrate compliance. Finally, detecting PII can support data minimization initiatives. Organizations often store personal data they do not actually need. Identifying this extraneous PII allows organizations to consider whether they can purge unnecessary data in order to further minimize privacy risks. In summary, PII detection provides vital visibility into sensitive data, bolsters data security and compliance, and enables minimizing retention of unneeded personal information.

## Installation
To install the package in an existing environment, run
```pip install -r requirements.txt```

## Dataset

The dataset is from the Kaggle competition "PII Data Detection" and contains ~22,000 essays written by students.

The data is in JSON format with the following fields:

- `document`: essay ID
- `full_text`: essay text 
- `tokens`: list of tokenized essay words/phrases
- `trailing_whitespace`  
- `labels`: BIO formatted tags (train set only)

The PII has been replaced by surrogates but is labeled as:

- `NAME_STUDENT`   
- `EMAIL`
- `USERNAME`
- `ID_NUM`  
- `PHONE_NUM`
- `URL_PERSONAL`
- `STREET_ADDRESS`

### Download the dataset
```kaggle competitions download -c pii-detection-removal-from-educational-data```

## TODOs

- [ ] Explore data format, PII labels.
- [ ] Split data into train/dev/test sets
- [ ] Research NER models (BERT, SpaCy, etc) 
- [ ] Fine-tune model on train data 
- [ ] Evaluate on dev set, tune hyperparameters
- [ ] Make predictions on test set
- [ ] Post-process predictions if needed
- [ ] Generate summary output for competition
- [ ] Write up methodology, results

## Links

- [Competition](https://www.kaggle.com/c/pii-detection-removal-from-educational-data)
- [Data](https://www.kaggle.com/c/pii-detection-removal-from-educational-data/data)

## Tests

To run tests  

On Windows:

```
$env:PYTHONPATH = "src"
pytest
```

On Linux:
```PYTHONPATH=src pytest```
