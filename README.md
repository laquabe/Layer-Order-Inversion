# Layer Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models
This is the source code for *Layer Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models*.

## Setup

Most environments can be installed:
```bash
 pip install -r requirements.txt
```

## Datasets
Please visit [Mquake](https://github.com/princeton-nlp/MQuAKE?tab=readme-ov-file) to download the datasets. We use MQuAKE-CF-3k-v2 as the basic data.

## Run

We recommend running `code/model_structure.py` first to examine the model structure, particularly the names of each layer.

### Data Split

run 
```
bash scripts/split_data.sh
``` 
to get the **Correct (Class 1), Incorrect (Class 3), and Missing (Class 4)** subsets.

### Patchscope

We follow the pipeline of [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/tree/main).

#### Data Preparation

- convert json to csv
    
    Run `code/patchscope/convert_json2csv.py` to convert mquake's JSON format to Patchscopes' input format CSV

- filter invalid line (optional)

    Before run Patchscopes, we recommend run `filter_valid_rows.sh` to filter the invalid line, especially for Llama3. Tokens within these rows may be untraceable, potentially causing runtime errors.

#### Run Patchscope

- Generation

    Pleas run the following script to get the generation results.
    ```
    bash scripts/patchscope.sh
    ```

- Filter
    To filter the random generation, please run `code/patchscope/filter_lowsim_generation.py`. You can set **global** or **case** as the filtering strategy.

#### Merge Results and Draw Pictures

- Merge

    To merge results, run the following script:
    ```
    bash scripts/merge_patchscopes_results.sh
    ```
    You will receive a JSON file containing the merge results and a CSV file containing summary statistics.

- Draw

    To draw the bar, run:
    ```
    python code/patchscope/summary_generation_results.py
    ```
    To draw the distribution line, run:
    ```
    bash scripts/draw_patchscopes_distribution.sh
    ```

### Hidden State Similarity

Before run, please check the layer name of the model.
Then run
```
bash scripts/hidden_state_sim.sh
```
You will get the similarity matrix(.npy) for each layer and the picture for one subset.

To merge results, please run

```
python code/hidden_state_similarity/merge_results.py
```
to get the comparsion figures.


