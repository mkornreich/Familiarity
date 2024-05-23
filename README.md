# Word Familiarity Calculator

A program that outputs familiarity values for workd.

## How it works

Uses data from the MRC Psycholinguistic Database as training data. Partitions the words from that data into two groups of words with high and low familiarity. Converts each of the words in the input to a corresponding vector in the GloVe dataset. Calculates the familiarity values of a word by evaluating the dot product of that word with each of the words in the groups of words with high and low familiarity.

## Usage

From a terminal

1. Clone this project `git clone
   https://github.com/mkornreich/Familiarity.git` and cd into it
   `cd Familiarity`
2. Download http://nlp.stanford.edu/data/wordvecs/glove.6B.zip and unzip it into the same directory
3. Install the necessary requirements
4. Run `python3 code.py`


