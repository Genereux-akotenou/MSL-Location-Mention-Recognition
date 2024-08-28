# Location Mention Recognition Challenge
This document outlines the approaches, models, and results obtained during the Location Mention Recognition challenge.

---

## 1. Introduction
The LMP systems consist of two main components i.e., LMR and LMD. The Location Mention Recognition (LMR) component is responsible for extracting toponyms, i.e., place or location names from microblogging posts and assigning location types (e.g., country, state, county, city) to them. Whereas, the Location Mention Disambiguation (LMD) component resolves the extracted location mentions to geographical areas using a geo-positioning database (i.e., gazetteer). In this particular challenge, we only focus on the Location Mention Recognition task.
The LMR task has aims at detecting location mentions and their spans regardless of their location types. Given a microblogging post, the task is to recognize all location mentions. For example, in the following microblogging post, “Paris” is the only location mentioned. Microblogging post: “Paris flooding shuts down train lines.”

---

## 2. Approach
- Try NER BERT-based model; Fine-tune LLM

---

## 3. Experiments and Models

### 3.1. Fine-Tuning BERT for NER
**Objective:**  
Fine-tune a pre-trained BERT model on the location mention recognition task.

**Model:**  
- **Architecture:** BERT: `bert-base-cased`
- **Training:** For parameters refer to notebook 2, 3, 4 in folder `approach_3`

**Steps:**
1. Load and preprocess the data.
2. Fine-tune the BERT model using the NERModel from `SimpleTransformers`.
3. Evaluate the model using appropriate metrics.

**Results:**
```markdown
| Metric        | Value  |
| ------------- | ------ |
| F1-Score      | XX.XX  |
| Precision     | XX.XX  |
| Recall        | XX.XX  |
| Accuracy      | XX.XX  |
```


### 3.2. Fine-Tuning LLama3.1
**Objective:**  
Explore the use of LLMs like LLaMA for the task of Location Mention Recognition.

**Model:**  
- **Architecture:** LLama3.1

**Steps:**
1. Fine tune like describe here https://medium.com/@nk94.nitinkumar/fine-tuning-large-language-models-for-named-entity-recognition-da97c38df742
2. Ref folder `approach5`


## 3. Extra consulted paper

- https://arxiv.org/pdf/2310.11324
- https://huggingface.co/docs/trl/sft_trainer
- https://medium.com/@nk94.nitinkumar/fine-tuning-large-language-models-for-named-entity-recognition-da97c38df742