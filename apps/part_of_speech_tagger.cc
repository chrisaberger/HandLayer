#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/logsoftmax.h"
#include "layers/lstm.h"
#include "tensor.h"
#include "utils/utils.h"

template <class Container>
void split1(const std::string& str, Container& cont) {
  std::istringstream iss(str);
  std::copy(std::istream_iterator<std::string>(iss),
            std::istream_iterator<std::string>(), std::back_inserter(cont));
}

std::unordered_map<std::string, size_t> load_text_file(
    const std::string& tag_file) {
  std::unordered_map<std::string, size_t> tag_to_ix;
  std::string line;
  size_t i = 0;
  std::ifstream myfile(tag_file);
  if (myfile.is_open()) {
    while (std::getline(myfile, line)) {
      if (!line.empty()) {
        tag_to_ix.insert({line, i++});
      }
    }
    myfile.close();
  } else
    std::cout << "Unable to open file";
  return tag_to_ix;
}

// TODO: Change this to read the default NLTK format.
std::vector<std::vector<std::pair<std::string, std::string>>> load_sentences(
    const std::string& tag_file) {
  std::string word;
  std::string token;
  std::vector<std::vector<std::pair<std::string, std::string>>> sentences;
  std::ifstream myfile(tag_file);
  std::vector<std::pair<std::string, std::string>> mypairs;
  if (myfile.is_open()) {
    while (std::getline(myfile, word)) {
      if (!word.empty()) {
        if (word == "end_sentence_done_here") {
          sentences.push_back(mypairs);
          mypairs.clear();
        } else {
          std::getline(myfile, token);
          mypairs.push_back(std::make_pair(word, token));
        }
      }
    }
    myfile.close();
  } else
    std::cout << "Unable to open file";
  return sentences;
}

std::pair<Tensor<size_t>, Tensor<size_t>> prepare_word_sequence(
    const std::vector<std::pair<std::string, std::string>>& sentence,
    const std::unordered_map<std::string, size_t>& word_to_ix,
    const std::unordered_map<std::string, size_t>& tag_to_ix) {
  std::vector<size_t> words;
  std::vector<size_t> tags;
  for (auto p : sentence) {
    const size_t word_ix = word_to_ix.at(std::get<0>(p));
    words.push_back(word_ix);
    const size_t tag_ix = tag_to_ix.at(std::get<1>(p));
    tags.push_back(tag_ix);
  }
  return std::make_pair(Tensor<size_t>::from_vec(words, {words.size()}),
                        Tensor<size_t>::from_vec(tags, {tags.size()}));
}

int main() {
  const size_t embedding_dim = 32;
  const size_t hidden_dim = 64;
  const size_t num_out_features = 12;

  LOG("LOADING EMBEDDING");
  Embedding<float> emb = Embedding<float>(
      Tensor<float>::from_npy("../apps/pos_data/embedding.npy"));

  LOG("LOADING LSTM");
  LSTM<float> lstm = LSTM<float>(embedding_dim, hidden_dim, 1, 1, true);
  lstm.set_weights(Tensor<float>::from_npy("../apps/pos_data/weight_ih_l0.npy"),
                   Tensor<float>::from_npy("../apps/pos_data/weight_hh_l0.npy"),
                   Tensor<float>::from_npy("../apps/pos_data/bias_ih_l0.npy"),
                   Tensor<float>::from_npy("../apps/pos_data/bias_hh_l0.npy"));

  LOG("LOADING LINEAR");
  Linear<float> linear(hidden_dim, num_out_features, true);
  linear.set_weights(
      Tensor<float>::from_npy("../apps/pos_data/linear_weight.npy"),
      Tensor<float>::from_npy("../apps/pos_data/linear_bias.npy"));

  Tensor<float> h = Tensor<float>::from_npy("../apps/pos_data/h_t.npy");
  Tensor<float> c = Tensor<float>::from_npy("../apps/pos_data/c_t.npy");
  h.shape = {1, hidden_dim};
  c.shape = {1, hidden_dim};

  LOG("LOADING TAG");
  const std::unordered_map<std::string, size_t> tag_to_ix =
      load_text_file("../apps/pos_data/tag_file.txt");
  
  LOG("LOADING WORDS");
  const std::unordered_map<std::string, size_t> word_to_ix =
      load_text_file("../apps/pos_data/word_file.txt");

  LOG("LOADING SENTENCES");
  const auto sentences = load_sentences("../apps/pos_data/test_sentences.txt");

  size_t hits = 0;
  size_t total = 0;
  size_t k = 0;
  LOG("Number of sentences: " + std::to_string(sentences.size()));
  for (auto sentence : sentences) {
    std::pair<Tensor<size_t>, Tensor<size_t>> w_t =
        prepare_word_sequence(sentence, word_to_ix, tag_to_ix);
    Tensor<size_t> word_ids = std::get<0>(w_t);
    Tensor<size_t> tag_ids = std::get<1>(w_t);
    Tensor<float> dec = emb.forward(word_ids);
    for (size_t i = 0; i < word_ids.numel; ++i) {
      Tensor<float> dec = emb.forward(word_ids.view(i, i + 1));
      auto h_c = lstm.forward(dec, h, c);
      h = std::get<0>(h_c);
      c = std::get<1>(h_c);

      Tensor<float> lin_out = linear.forward(std::get<0>(h_c));
      Tensor<float> out = LogSoftmax<float>::forward(lin_out);
      // out.print();
      float max = -std::numeric_limits<float>::infinity();
      size_t max_index = 0;
      for (size_t j = 0; j < out.numel; ++j) {
        if (out[j] > max) {
          max = out[j];
          max_index = j;
        }
      }
      hits += (max_index == tag_ids[i]);
      total++;
    }
  }
  std::cout << "Accuracy: " << 100 * (((float)hits) / ((float)total)) << " ("
            << hits << "/" << total << ")" << std::endl;
  return 0;
}