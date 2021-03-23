This repo contains my submission to the Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge 2020 and my supporting work. This competition, sponsored by the Radboud University Medical Center and Karolinska Institute, sought to drive the creation of the first ML/DL powered system for grading tissue biopsy images for prostate cancer (PCa) detection at or above pathologist-level accuracy on a large dataset including samples collected from more than one research center. 

Challenges:
- Prior attempts had been limited in scope to smaller datasets from single research centers
- Variance exists between location specific datasets in the processes that they use during biopsy and slide preparation (In this case we can see significant variation in the coloring of slides between Radboud and Karolinska)
- Variance exists in the grading between specific pathologists (here we see both different labeling methods between the pathologists at the two centers and a different distribution of scores between the two)
- Whole-slide images are very large at full resolution, frequently over 10,000 pixels per dimension
- These images also typically contain large amounts of blank space due to the varied sizes and shapes of tissue samples, this white space is obviously computationally expensive to process without adding any value to the diagnosis

In order to follow my thought process throughout the challenge I recommend viewing the notebooks in the following order:
1. [Exploratory Data Analysis (panda-eda.ipynb)](https://github.com/kittmiller25/kaggle_pCancer_comp/blob/main/panda-eda.ipynb "EDA notebook")
2. [Approach #1 - Pseudo Segmentation (panda-pseudosegmentation.ipynb)](https://github.com/kittmiller25/kaggle_pCancer_comp/blob/main/panda-pseudosegmentation.ipynb "PS notebook")
3. [Approach #2 - End-to-End: Image Preparation (panda-image-extraction.ipynb)](https://github.com/kittmiller25/kaggle_pCancer_comp/blob/main/panda-image-extraction.ipynb "Image extraction notebook")
4. [Approach #2 - End-to-End: Modeling (panda-endtoend_v1.ipynb)](https://github.com/kittmiller25/kaggle_pCancer_comp/ "Modeling notebook")
5. [Inference Testing (panda_inference_test.ipynb)](https://github.com/kittmiller25/kaggle_pCancer_comp/ "Inference notebook")

Result - I achieved a final score (quadratic weighted kappa score) of 0.898 vs the competition winner at 0.941


```python

```
