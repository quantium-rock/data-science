# ⛰ "Train At Scale" Unit 🗻

In this unit, you will learn how to package the notebook provided by the Data Science team at WagonCab, and how to scale it so that it can be trained on the full dataset locally on your machine.

This unit consist of the 5 challenges below, that are all regrouped in this single README file.

Simply follow the guide and `git push` after each main section so we can track your progress!

# 1️⃣ LOCAL SETUP

<details>
  <summary markdown='span'><strong>❓ instructions (expand me)</strong></summary>

As lead ML Engineer for the project, your first role is to setup a local working environment (pyenv) and a python package that only contains the skeleton of your code base.

💡 Packaging notebooks is a key ML Engineer skill. It allows
- Other users to collaborate on the code
- To call the code locally or on a remote machine in order for example to train the `taxifare` on a bigger machine
- To put the code in production (on a server that never stops running) in order to expose it as an **API** or through a **website**
- Render the code operable so that it can be ran manually or plugged to an automation workflow

### 1.1) Create new pyenv [🐍 taxifare-env]

❓ Create the virtual env

```bash
cd ~/code/<user.github_nickname>/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
python --version # First, check your <YOUR_PYTHON_VERSION>. For example: 3.8.12
```

```bash
pyenv virtualenv <YOUR_PYTHON_VERSION> taxifare-env
pip install --upgrade pip
pyenv local taxifare-env
code .
```

Then, make sure both your OS terminal, your VS-code integrated terminal display well [🐍 taxifare-env].
On VS code, open any `.py` file and check that taxifare-env is also activated by clicking on the bottom right pyenv section as below
<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-setup.png'>

### 1.2) Get familiar with the taxifare package stucture

❓ Take 10 min to understand the structure of the boilerplate we've prepared for you (don't go into detai. It's entry is `taxifare.interface.main_local`

```bash
. # Challenge folder root
├── Makefile          # Main "interface" with your project. Use to launch tests, or start trainings etc... from the CLI
├── README.md         # The file you are reading right now!
├── notebooks
│   └── datascientist_deliverable.ipynb # The deliverable from the DS team!
├── pytest.ini        # test configuration file (do not touch)
├── requirements.txt  # list all third party packages to add to your local environment
├── setup.py          # allow to `pip install` your package
├── taxifare          # the code logic for this package
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py   # Your main python entry point that contains all the "routes" that will be accessible from "outside"
│   └── ml_logic
│       ├── __init__.py
│       ├── data.py         # save, load and clean data
│       ├── encoders.py     # custom encoders utilities
│       ├── model.py        # tensorflow model
│       ├── params.py       # global project params
│       ├── preprocessor.py # sklearn preprocessing pipelines
│       ├── registry.py     # save and load models
│       └── utils.py        # useful python functions that can be shared accross the taxifare package
├── tests  # Tests to run using make pytest
│   ├── ...
│   └── ...
├── .gitignore
```

❓ Install your package on this new virtual env.

```bash
cd ~/code/<user.github_nickname>/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
pip install -e .
```

Make sure the package is installed by running `pip list | grep taxifare`. It should print the absolute path to the package.


### 1.3) Let's store all our data locally at `~/.lewagon/mlops/`

❓ Let's store our `data` folder *outside* of this challenge folder, so that it can be accessible by all other challenges during the whole MLOPS module. We don't want it to be git tracked anyway!

``` bash
mkdir -p ~/.lewagon/mlops/data
mkdir -p ~/.lewagon/mlops/data/raw
mkdir -p ~/.lewagon/mlops/data/processed
```

❓ While we are here, let's also create a storage folder for our `training_outputs` that will also be shared by all challenges

```bash
mkdir -p ~/.lewagon/mlops/training_outputs
mkdir -p ~/.lewagon/mlops/training_outputs/metrics
mkdir -p ~/.lewagon/mlops/training_outputs/models
mkdir -p ~/.lewagon/mlops/training_outputs/params
```

You can now see that the data for the challenges to come is stored in `~/.lewagon/mlops/` along with the notebooks of the data science team and the model outputs:

``` bash
tree -a ~/.lewagon/mlops/

# YOU SHOULD SEE THIS
├── data              # This is where
│   ├── processed     # You will store intermediate processed data
│   └── raw           # You will download samples of the raw data
└── training_outputs
    ├── metrics       # trained model metrics
    ├── models        # trained model weights (can be large!)
    └── params        # trained model hyper parameters
```

❓ Now, download the raw datasets

```bash
# 4 train sets
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/train_1k.csv > ~/.lewagon/mlops/data/raw/train_1k.csv
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/train_10k.csv > ~/.lewagon/mlops/data/raw/train_10k.csv
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/train_100k.csv > ~/.lewagon/mlops/data/raw/train_100k.csv
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/train_500k.csv > ~/.lewagon/mlops/data/raw/train_500k.csv

# 4 val sets
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/val_1k.csv > ~/.lewagon/mlops/data/raw/val_1k.csv
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/val_10k.csv > ~/.lewagon/mlops/data/raw/val_10k.csv
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/val_100k.csv > ~/.lewagon/mlops/data/raw/val_100k.csv
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/val_500k.csv > ~/.lewagon/mlops/data/raw/val_500k.csv
```

❗️ And only if you have excellent internet connexion and 6Go free space on your computer (not mandatory)

```bash
curl https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/train_50M.csv.zip > ~/.lewagon/mlops/data/raw/train_50M.csv.zip
```

</details>

# 2️⃣ UNDERSTAND DATA SCIENTIST WORK

<details>
  <summary markdown='span'><strong>❓ instructions (expand me)</strong></summary>

*⏱ Duration:  spend 1 hour max on this*

❓ Open `datascientist_deliverable.ipynb` within VScode (forget about Jupyter for this module), and run all cells carefully while understanding them. This handover between you and the DS team is the perfect time to interact with them (i.e your buddy or a TA).

❗️ Make sure to use `taxifare_model` as ipykernel venv

<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-notebook.png' width=400>

</details>


# 3️⃣ PACKAGE CODE

<details>
  <summary markdown='span'><strong>❓ instructions (expand me)</strong></summary>

🎯 Your goal is to be able to run the `taxifare.interface.main_local` module as per below

```bash
# -> model
python -m taxifare.interface.main_local
```

❓ To do so, please code the missing code marked `# YOUR CODE HERE` in the following files. It should follow the Notebook pretty closely!

```markdown
├── taxifare
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py   # ❓ 💡 Start here: code `preprocess_and_train`, `pred`
│   └── ml_logic
│       ├── __init__.py
│       ├── data.py         # ❓ `clean data`
│       ├── encoders.py     # ❓ `transform_time_features`, `transform_lonlat_features`, `compute_geohash`
│       ├── model.py        # ❓ `initialize_model`, `compile_model`, `train_model`
│       ├── params.py       # ✅ You can change `DATASET_SIZE`
│       ├── preprocessor.py # ❓ `preprocess_features`
│       ├── registry.py     # ✅ `save_model` and `load_model` are already coded for you
│       └── utils.py        # ✅ keep for later
```

**🧪 Test your code**

❓ First, make sure your package runs properly with `python -m taxifare.interface.main_local`.
- Debug it until it runs!
- Use the following dataset size

```python
# taxifare/ml_logic/params.py
DATASET_SIZE = '1k' # To iterate faster in debug mode 🐞
DATASET_SIZE = '100k' # Should work at least once with it
```

❓ Then only, try to pass tests with `make test_train_at_scale_3`


</details>

# 4️⃣ INVESTIGATE SCALABILITY

<details>
  <summary markdown='span'><strong>❓ instructions (expand me)</strong></summary>

*⏱ Duration:  spend 20 minutes max on this here*

Now that you managed to make the package work for a small dataset, time to see how it will handle the real dataset!

👉 Switch `ml_logic.params.DATASET_SIZE` and `ml_logic.params.VALIDATION_DATASET_SIZE` to `'500k'` to start getting serious!

❓ Investigate **which part of your code takes the most time and memory usage** and try to answer the following questions with your buddy:
- [ ] What part of your code holds the key bottlenecks?
- [ ] What kind of bottlenecks are the most worrying? (Time, Memory?)
- [ ] Do you think it will scale to 50M rows?
- [ ] Can you think about potential solutions? Write down your ideas, but do not implement them yet!

💡 Hint: Use `ml_logic.utils.simple_time_and_memory_tracker` to decorate the methods of your choice as below

```python
# taxifare.ml_logic.data.py
from taxifare.ml_logic.utils import simple_time_and_memory_tracker

@simple_time_and_memory_tracker
def clear_data() -> pd.DataFrame:
    ...
```

(💡 Optional) If you don't remember exactly how decorators work, refer to our lecture [0405-Communicate](https://kitt.lewagon.com/camps/<user.batch_slug>/lectures/content/04-Decision-Science_05-Communicate.slides.html?title=Communicate#/6/3)

</details>


# 5️⃣ INCREMENTAL PROCESSING

<details>
  <summary markdown='span'><strong>❓ instructions (expand me)</strong></summary>

🎯 Your goal is to improve your codebase so as **to be able to process our model on `50M` rows or even more, without RAM limits**.

### 5.1) Discussion

**What did we learn?**

From previous challenge, we've seen that we have memory and time constraints:
- the `(55M,8)` `raw_data` loaded in memory as dataframe takes about 12GB of RAM, which is too much for most computers.
- the `(55M,65)` preprocessed dataframe is even bigger.
- the `ml_logic.encoders.compute_geohash` method takes an awful long time to process 🤯

**What could we do?**

1. One solution is to buy **more RAM from a Virtual Machine** in the cloud and process it there (and it is often the simplest way to deal with such problem)
2. Another could be to load each column of the `raw_data` individually, and prepare some preprocessing on it, **column by column**
```python
for col in column_names:
    df_col = pd.read_csv("raw_data.csv", usecols=col)
    # do preprocessing on the single column here
```

However, you may encounter datasets whose individual columns are "too big to load anyway"! By the way, the [real NYC dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) is even bigger than 55M rows and actually weight about 156GB !

**Proposed solution: incremental preprocessing 🔪 chunk-by-chunk 🔪**

Did you notice our preprocessing is **stateless**?
- We don't need to store (_fit_) any information about columns of the train set (such as _standard deviation_), to apply it (_transform_) on the test set.
- We can therefore decouple the _preprocessing_ from the _training_ instead of grouping everything into a pipeline `preprocess_and_train`.
  - We will `preprocess` and store `data_processed` once-for-all on our hard drive
  - Then `train` our model from that `data_processed` later on.
  - When new data will arrive, we'll simply apply the preprocessing to it as a pure python function.

Secondly, as we do not need to compute _column-wise-statistics_ but only perform _row-by-row preprocessing_, we can do the preprocessing **chunk by chunk**, with chunks of limited size (e.g 100_000 rows), each chunk fitting nicely in memory! And then simply append each _processed chunk_ at the end of a CSV on our local disk. It won't make it faster but at least it will compute without crashing. And you only need to do it once.

<img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/process_by_chunk.png">

### 5.2) Your turn

❓ **First, bring back smaller dataset sizes for debug purpose**

```python
# params.py
DATASET_SIZE = '1k'
VALIDATION_DATASET_SIZE = '1k'
CHUNK_SIZE = 200
```

❓ **Then, copy-paste and code this new route given below `def preprocess()` in your `ml_logic.interface.main_local` module**

[//]: # (  🚨 Code below is NOT the single source of truth. Original is in data-solutions repo 🚨 )

<br>

<details>
  <summary markdown='span'>👇 Code to copy 👇</summary>

```python
def preprocess(source_type='train'):
    """
    Preprocess the dataset iteratively, loading data by chunks fitting in memory,
    processing each chunk, appending each of them to a final dataset preprocessed,
    and saving final prepocessed dataset as CSV
    """

    print("\n⭐️ use case: preprocess")

    # local saving paths given to you (do not overwrite these data_path variable)
    source_name = f"{source_type}_{DATASET_SIZE}.csv"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}.csv"

    data_raw_path = os.path.abspath(os.path.join(LOCAL_DATA_PATH, "raw", source_name))
    data_processed_path = os.path.abspath(os.path.join(LOCAL_DATA_PATH, "processed", destination_name))

    # iterate on the dataset, by chunks
    chunk_id = 0

    # Let's loop until we reach the end of the dataset, then `break` out
    while (True):
        print(f"processing chunk n°{chunk_id}...")

        try:
            # load in memory the chunk numbered `chunk_id` of size `CHUNK_SIZE`
            # 🎯 Hint: check out pd.read_csv(skiprows=..., nrows=..., headers=...)
            # We advise you to always load data with `header=None`, and add back column names using COLUMN_NAMES_RAW
            # 👉 YOUR CODE HERE

        except pd.errors.EmptyDataError:
            # 🎯 Hint: What would you do when you reached the end of the CSV ?
            # 👉 YOUR CODE HERE


        # clean chunk. Pay attention, sometimes it can result in 0 rows remaining!
        # 👉 YOUR CODE HERE

        # create X_chunk,y_chunk
        # 👉 YOUR CODE HERE

        # create X_processed_chunk and concatenate (X_processed_chunk, y_chunk) into data_processed_chunk
        # 👉 YOUR CODE HERE

        # Save and append the chunk of the preprocessed dataset to a local CSV
        # Keep headers on the first chunk: For convention, we'll always save CSVs with headers in this challenge
        # 🎯 Hints: check out pd.to_csv(mode=...)

        # 👉 YOUR CODE HERE

        chunk_id += 1

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    data_processed = pd.read_csv(data_processed_path, header=None, skiprows=1, dtype=DTYPES_PROCESSED_OPTIMIZED).to_numpy()
    write_result(name="test_preprocess", subdir="train_at_scale", data_processed_head=data_processed[0:10])


    print("✅ data processed saved entirely")

```

</details>

<br>

**❓ Try create and store the following preprocessed datasets**

- `data/processed/train_processed_1k.csv` by running `preprocess()`
- `data/processed/val_processed_1k.csv` by running `preprocess(source_type='val')`

**🧪 Test your code**

Test your code with `make test_train_at_scale_5`.

**❓ Finally, create and store the real preprocessed datasets**

Using:
```python
# params.py
DATASET_SIZE = '500k'
VALIDATION_DATASET_SIZE = '500k'
CHUNK_SIZE = 100000
```
To create:
- `data/processed/train_processed_500k.csv` by running `preprocess()`
- `data/processed/val_processed_500k.csv` by running `preprocess(source_type='val')`

🎉 Given few hours of computation, we could easily process the 55 Millions rows too, but let's not do it today!

</details>

# 6️⃣ INCREMENTAL LEARNING

<details>
  <summary markdown='span'><strong>❓ instructions (expand me)</strong></summary>

🎯 Goal: Train our model on the full `data_processed.csv`

### 6.1) Discussion

We cannot load such dataset of shape (55M, 65) in RAM all at once, but we can load it chunk by chunk.

**How do we train a model "chunk by chunk" ?**

This is called **incremental learning** or **partial_fit**
- We initialize a model with random weights ${\theta_0}$
- We load the first `data_processed_chunk` in memory (say, 100_000 rows)
- We train model on the first chunk, and update its weights accordingly ${\theta_0} \rightarrow {\theta_1}$
- We load the second `data_processed_chunk` in memory
- We *retrain* model with this second chunk, this time updating previously computed weights ${\theta_1} \rightarrow {\theta_2}$!
- etc... until the end of the entire dataset

❗️ Not all machine-learning model support incremental learning: only *parametric* models $f_{\theta}$ that are based on *iterative update methods* like gradient descent do
- In **scikit-learn**, `model.partial_fit()` is only available SGDRegressor/Classifier and few others ([read this carefully 📚](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning)).
- In **tensorflow** and another other deep learning framework, training is always iterative and incremental learning is the default behavior! You just need to avoid calling `model.initialize()` between two chunks!

❗️ Do not confuse `chunk_size` with `batch_size` from deep learning
- For each chunk (big), your model will read data batch-per-batch (small) many times over (epochs)

<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/train_by_chunk.png'>


👍 **Pros:**: This universal approach is framework independent. You can use it with scikit-learn, XGBoost, Tensorflow etc...

👎 **Cons**: The model will be biased towards fitting the *latest chunk* better than the *first* ones. In our case, it is not a problem as our training dataset is shuffled, but it is important to keep that in mind when we will do a partial-fit of our model with newer data once it is in production.

<br>

<details>
  <summary markdown='span'><strong>🤔 Do we really need chunks with tensorflow?</strong></summary>

Granted, thanks to tensorflow `Datasets` you will not always need "chunks" as you can use batch-per-batch dataset loading as below

```python
import tensorflow as tf
ds = tf.data.experimental.make_csv_dataset(data_processed_55M.csv, batch_size=256)
model.fit(ds)
```

However, we would like to teach you the universal method of incremental fit by chunk in this challenge, as it applies to any framework, and will prove useful to *partially retrain* your model with newer data once it is put in production.
</details>

<br>

### 6.2) Your turn

**❓ Copy paste and try to code this new route `def train()` below in your `ml_logic.interface.main_local` module**

(Again, start with super-small dataset size, then finally train your model on 500k)

[//]: # (  🚨 Code below is not the single source of truth 🚨 )

<details>
  <summary markdown='span'><strong>👇 Code to copy 👇</strong></summary>

```python
def train():
    """
    Training on the full (already preprocessed) dataset, by loading it
    chunk-by-chunk, and updating the weight of the model for each chunks.
    Save model, compute validation metrics on a holdout validation set that is
    common to all chunks.
    """
    print("\n ⭐️ use case: train")

    # Validation Set: Load a validation set common to all chunks and create X_val, y_val
    data_val_processed_path = os.path.abspath(os.path.join(
        LOCAL_DATA_PATH, "processed", f"val_processed_{VALIDATION_DATASET_SIZE}.csv"))

    data_val_processed = pd.read_csv(
        data_val_processed_path,
        skiprows= 1, # skip header
        header=None,
        dtype=DTYPES_PROCESSED_OPTIMIZED
        ).to_numpy()

    X_val = data_val_processed[:, :-1]
    y_val = data_val_processed[:, -1]

    # Iterate on the full training dataset chunk per chunks.
    # Break out of the loop if you receive no more data to train upon!
    model = None
    chunk_id = 0
    metrics_val_list = []  # store each metrics_val_chunk

    while (True):
        print(f"loading and training on preprocessed chunk n°{chunk_id}...")

        # Load chunk of preprocess data and create (X_train_chunk, y_train_chunk)
        path = os.path.abspath(os.path.join(
            LOCAL_DATA_PATH, "processed", f"train_processed_{DATASET_SIZE}.csv"))

        try:
            data_processed_chunk = pd.read_csv(
                    path,
                    skiprows=(chunk_id * CHUNK_SIZE) + 1, # skip header
                    header=None,
                    nrows=CHUNK_SIZE,
                    dtype=DTYPES_PROCESSED_OPTIMIZED,
                    ).to_numpy()

        except pd.errors.EmptyDataError:
            data_processed_chunk = None  # end of data

        # Break out of while loop if we have no data to train upon
        if data_processed_chunk is None:
            break

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]

        learning_rate = 0.001
        batch_size = 256
        patience = 2

        # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
        # 👉 YOUR CODE HERE

        chunk_id += 1

    # return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience = patience,
        incremental=True,
        chunk_size=CHUNK_SIZE)

    print(f"\n✅ trained with MAE: {round(val_mae, 2)}")

    save_model(model, params=params, metrics=dict(mae=val_mae))

    print("✅ model trained and saved")

```

</details>

**🧪 Test your code with `make test_train_at_scale_6`**

You should get an MAE below 3 on the validation set!

🏁 🏁 🏁 🏁 Congratulations! 🏁 🏁 🏁 🏁

</details>
