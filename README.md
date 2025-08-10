# data-preprocessing
It’s the step where we prepare raw data so our machine learning model can understand it better. Most raw data is messy — it can have missing values, wrong formats, or numbers in very different ranges.  
 Main steps in data preprocessing
1️⃣ Handling Missing Values
Use SimpleImputer or KNNImputer to fill missing data.

Options: fill with mean, median, most frequent, or predicted values.

2️⃣ Encoding Categorical Data
Many datasets have text values (like "Male", "Female").

Convert text to numbers using:

Label Encoding → "Male"=0, "Female"=1

One-Hot Encoding → creates columns like Male=1, Female=0

3️⃣ Scaling/Normalization
Make sure all numeric columns have similar ranges:

Min–Max Scaling → values between 0 and 1

Standardization (Z-score) → mean = 0, std = 1

4️⃣ Feature Selection/Engineering
Choose only the most useful features for the model.

Create new features from existing ones (e.g., "Age Group" from "Age").

5️⃣ Splitting Data
Use train_test_split from sklearn to split into:

Training data → to teach the model

Test data → to check if the model learned well
