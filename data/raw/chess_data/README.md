Included is data from Google DeepMind's `Searchless Chess` project. See [repo](https://github.com/google-deepmind/searchless_chess/blob/main/LICENSE). Our *train*, *test* and *evals* are all from this source.

This data leverages the 62k samples from the 'Action-Value Test Dataset'. If you intend to generate more data (e.g., we generated millions of samples in our training) we recommend forking their repository ([link](https://github.com/google-deepmind/searchless_chess/blob/main/README.md)), downloading the train 'Behavioral Cloning' dataset, processing the '.bag' file, and having a csv with the headings 'FEN', 'Best Move' from this dataset.

We include 100k samples in the `deepmind_behaviorcloning_100k.csv` file.