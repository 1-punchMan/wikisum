# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wikipedia Summarization Problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import re
import string
import tempfile
import time, requests, subprocess, pickle, fastBPE, random, numpy as np
from numpy import linalg as la

import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators.wikisum import utils as cc_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf
from urllib.parse import quote

import torch
from torch.utils.data import Dataset as Torch_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

#XLM
import sys
sys.path.append("/home/zchen/XLM/src/data/")
from dictionary import Dictionary

PROCESS_FOLDER_PREFIX = "process"
REF_SHARD_FILE_PREFIX = "references.tfrecords.gz"
REF_SHARD_FILE = REF_SHARD_FILE_PREFIX + "-%05d-of-01000"

# Support files
BASE_SUPPORT_DIR = "gs://tensor2tensor-data/wikisum"
WIKI_CONTENT_DIR = os.path.join(BASE_SUPPORT_DIR, "wiki_content")
WIKI_URLS_DIR = os.path.join(BASE_SUPPORT_DIR, "wiki_urls")
WET_METADATA_DIR = os.path.join(BASE_SUPPORT_DIR, "commoncrawl_metadata")
WIKI_CONTENT_FILE = "wiki_content.tfrecords-%05d-of-01000"
WIKI_URLS_FILE = "wiki_urls.json-%05d-of-01000"
LINKED_ARTICLES_FILE="iisr-wikisum-bucket2/linked_articles/linked_articles-{:0>5d}-of-01000.json"

EOT = "<EOT>"  # end-of-title string
_MIN_REFS = 1
_MIN_LEADSECTION_TOKENS = 1


class WikisumBase(problem.Problem):
  """Base class for Wikisum problems."""

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "section_boundaries": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  @property
  def target_vocab_size(self):
    return 2**15

  @property
  def vocab_filename(self):
    return "vocab.%s.%d" % (self.dataset_filename(), self.target_vocab_size)

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    # Shared encoder for inputs and targets
    return {"inputs": encoder, "targets": encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = True

    p.vocab_size = {
        "inputs": self._encoders["inputs"].vocab_size,
        "targets": self._encoders["targets"].vocab_size,
    }
    p.modality = {
        "inputs": modalities.ModalityType.SYMBOL,
        "targets": modalities.ModalityType.SYMBOL,
    }

  def eval_metrics(self):
    return super(WikisumBase, self).eval_metrics() + [
        metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
    ]

  def generate_lines_for_vocab(self, wikis_dir, refs_dir, max_chars=10**7):
    total_chars = 0
    ref_files_by_shard = _references_files_by_shard(refs_dir)
    for shard_id in range(cc_utils.NUM_SHARDS):
      # Wikipedia articles
      for wiki in _wiki_articles(shard_id, wikis_dir):
        yield _normalize_text(wiki.title) + EOT
        for section in wiki.sections:
          yield _format_title(_normalize_text(section.title))
          yield _normalize_text(section.text)
          total_chars += len(section.title)
          total_chars += len(section.text)

      # References
      for i, content in enumerate(
          six.itervalues(_references_content(ref_files_by_shard[shard_id]))):
        for line in content.split("\n"):
          if line:
            yield _normalize_text(line)
            total_chars += len(line)

        # Make sure we use at least 1k references
        if i >= 1000 and total_chars >= max_chars:
          break

      if total_chars >= max_chars:
        tf.logging.info("Seen enough chars: %d; finished.", max_chars)
        break
    tf.logging.info("Built vocabulary using %d chars", total_chars)

  def generate_vocab(self, data_dir, wikis_dir, refs_dir):
    # Produce a SubwordTextEncoder from a subset of the data
    return generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_filename, self.target_vocab_size,
        self.generate_lines_for_vocab(wikis_dir, refs_dir))

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warn("See wikisum/README.md for instructions to generate data.")

  def out_filepaths(self, data_dir):
    train_shards = 800
    dev_shards = 100
    test_shards = 100
    train_filepaths = self.training_filepaths(
        data_dir, train_shards, shuffled=True)
    dev_filepaths = self.dev_filepaths(data_dir, dev_shards, shuffled=True)
    test_filepaths = self.test_filepaths(data_dir, test_shards, shuffled=True)
    out_filepaths = train_filepaths + dev_filepaths + test_filepaths
    out_filepaths.sort()
    assert len(out_filepaths) == cc_utils.NUM_SHARDS
    return out_filepaths


@registry.register_problem
class WikisumCommoncrawl(WikisumBase):
  """Wikipedia references->article summarization task based on CommonCrawl."""
  pass


@registry.register_problem
class WikisumWeb(WikisumBase):
  """Wikipedia references->article summarization task based on web data."""
  pass


@registry.register_problem
class WikisumCommoncrawlLeadSection(WikisumCommoncrawl):
  """Wikipedia references->lead section summarization task."""

  def preprocess_example(self, example, mode, hparams):
    example["targets"] = _truncate_to_lead_section(example)
    return super(WikisumCommoncrawlLeadSection, self).preprocess_example(
        example, mode, hparams)

  def dataset_filename(self):
    return WikisumCommoncrawl.name

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warn("Problem %s reuses data from problem %s", self.name,
                    WikisumCommoncrawl.name)


@registry.register_problem
class WikisumWebLeadSection(WikisumWeb):
  """Wikipedia references->lead section summarization task."""

  def preprocess_example(self, example, mode, hparams):
    example["targets"] = _truncate_to_lead_section(example)
    return super(WikisumWebLeadSection, self).preprocess_example(
        example, mode, hparams)

  def dataset_filename(self):
    return WikisumWeb.name

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warn("Problem %s reuses data from problem %s", self.name,
                    WikisumWeb.name)


def make_ref_shard_files(out_dir):
  tf.gfile.MakeDirs(out_dir)
  opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
  files = [
      tf.python_io.TFRecordWriter(
          os.path.join(out_dir, REF_SHARD_FILE % i), opts)
      for i in range(cc_utils.NUM_SHARDS)
  ]
  return files


def _truncate_to_lead_section(example):
  wiki = example["targets"]
  lead_boundary = example["section_boundaries"][0]
  # Concat a new EOS to the lead since the original one gets truncated.
  lead = tf.concat((wiki[:lead_boundary], [text_encoder.EOS_ID]), 0)
  return lead


def _make_example_from_record(record):
  features = {
      "url":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[record.url])),
      "content":
          tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[record.content])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def _shard_id_for_file(sharded_filename):
  suffix = "00000-of-00000"
  parts = sharded_filename[-len(suffix):].split("-")
  assert len(parts) == 3
  return int(parts[0])


def _references_files_by_shard(refs_dir):
  process_dirs = _process_folders(refs_dir)
  shards = collections.defaultdict(list)
  for d in process_dirs:
    ref_files = tf.gfile.Glob(os.path.join(d, REF_SHARD_FILE_PREFIX) + "*")
    for f in ref_files:
      shards[_shard_id_for_file(f)].append(f)
  return shards


def _references_content(ref_files):
  """Returns dict<str ref_url, str ref_content>."""
  example_spec = {
      "url": tf.FixedLenFeature([], tf.string),
      "content": tf.FixedLenFeature([], tf.string),
  }
  data = {}
  for ex in generator_utils.tfrecord_iterator(
      ref_files, gzipped=True, example_spec=example_spec):
    data[ex["url"]] = text_encoder.to_unicode(ex["content"])
  return data


def _wiki_urls_for_shard(shard_id, urls_dir=None):
  """Urls for chunk: dict<str wiki_url, list<str> ref_urls>."""
  urls_dir = urls_dir or WIKI_URLS_DIR
  urls_filepath = os.path.join(urls_dir, WIKI_URLS_FILE % shard_id)
  with tf.gfile.GFile(urls_filepath) as f:
    return json.loads(f.read())


class WikipediaSection(
    collections.namedtuple("WikipediaSection", ["title", "text"])):
  pass


class WikipediaArticle(
    collections.namedtuple("WikipediaArticle", ["url", "title", "sections"])):
  pass


def _wiki_articles(shard_id, wikis_dir=None):
  """Generates WikipediaArticles from GCS that are part of shard shard_id."""
  if not wikis_dir:
    wikis_dir = WIKI_CONTENT_DIR
  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(
        cc_utils.readahead(
            os.path.join(wikis_dir, WIKI_CONTENT_FILE % shard_id)),
        buffer_size=16 * 1000 * 1000)

    def _parse_example(ex_ser):
      """Parse serialized Example containing Wikipedia article content."""
      features = {
          "url": tf.VarLenFeature(tf.string),
          "title": tf.VarLenFeature(tf.string),
          "section_titles": tf.VarLenFeature(tf.string),
          "section_texts": tf.VarLenFeature(tf.string),
      }
      ex = tf.parse_single_example(ex_ser, features)
      for k in ex.keys():
        ex[k] = ex[k].values
      ex["url"] = ex["url"][0]
      ex["title"] = ex["title"][0]
      return ex

    dataset = dataset.map(_parse_example, num_parallel_calls=32)
    dataset = dataset.prefetch(100)
    record_it = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      while True:
        try:
          ex = sess.run(record_it)
        except tf.errors.OutOfRangeError:
          break

        sections = [
            WikipediaSection(title=text_encoder.to_unicode(title),
                             text=text_encoder.to_unicode(text))
            for title, text in zip(ex["section_titles"], ex["section_texts"])
        ]
        yield WikipediaArticle(
            url=text_encoder.to_unicode(ex["url"]),
            title=text_encoder.to_unicode(ex["title"]),
            sections=sections)

def generate_wiki_data(shard_id, wikis_dir, from_WikiAPI=False):
  file=LINKED_ARTICLES_FILE.format(shard_id)
  with tf.gfile.Open(file) as f:
      linked_articles = json.load(f)
      
  file=f"/home/zchen/XLM/wikisum/en2zh_dataset/zh_articles/zh_articles-{shard_id:>05}-of-01000.json"
  with tf.gfile.Open(file) as f:
      zh_articles = json.load(f)

  last=len(linked_articles)-1
  articles={}
  for en_wiki in _wiki_articles(shard_id, wikis_dir):
    if en_wiki.url not in linked_articles: continue
    zh_title = linked_articles[en_wiki.url]
    if from_WikiAPI:  pass
      # 待改寫
      # articles[title]=url
      # if (i+1)%50 == 0 or i == last:
      #   titles_str='|'.join([quote(title) for title in articles])
      #   api=f"https://zh.wikipedia.org/w/api.php?action=query&titles={titles_str}&prop=extracts&explaintext&exintro&utf8=&format=json"
        
      #   #wiki的server偶爾會出錯，需多次嘗試
      #   while True:
      #       try:
      #           r = requests.get(api)
      #           if r.status_code == 200: break
      #       except OSError:
      #           pause_time=1
      #           print()
      #           print("ConnectionError")
      #           print(f"等{pause_time}秒......")
      #           print()
      #           time.sleep(pause_time)
                
      #   items=json.loads(r.text)["query"]["pages"]
      #   for item in items.values():
      #     zh_title=item.get("title")
      #     intro=item.get("extract")
      #     url=articles.get(zh_title)
      #     if intro is None or url is None: continue

      #     yield url, zh_title, intro

      #   articles={}

      #   time.sleep(0.5)
    else:
      zh_intro = zh_articles.get(zh_title)
      if zh_intro is None: continue

      yield en_wiki, zh_title, zh_intro

def _token_counts(text, token_set=None):
  counts = collections.defaultdict(int)
  for token in tokenizer.encode(text_encoder.native_to_unicode(text)):
    if token_set and token not in token_set:
      continue
    counts[token] += 1
  return counts


def _normalize_text(text):
  text = text.lower()
  # Space around punctuation
  text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def _tokens_to_score(tokens):
  return {t for t in tokens if re.search("[a-z0-9]", t)}

class Ranking_Dataset(Torch_dataset):

    def __init__(self, zh_intro, ref, maxlen, dico):
        zh_intro = np.concatenate(zh_intro)

        self.inputs = ref + [zh_intro]
        self.maxlen=maxlen
        self.dico=dico
        
    def __getitem__(self, index):
        eos_index=self.dico.eos_index
        paragraph = [eos_index] + self.inputs[index].astype(np.int64).tolist() + [eos_index]
        paragraph = paragraph[:self.maxlen]
        langs = index == len(self.inputs) - 1

        return paragraph, len(paragraph), langs

    def __len__(self):
        return len(self.inputs)

def collate(samples, padding_value):
    inputs = [s[0] for s in samples]
    inputs=[torch.LongTensor(t) for t in inputs]
    inputs = pad_sequence(inputs, padding_value=padding_value)    # (slen, bs)
    
    ilen = [s[1] for s in samples]
    ilen = torch.LongTensor(ilen)
    langs = [s[2] for s in samples]
    langs = torch.LongTensor(langs).unsqueeze(0).expand_as(inputs)
    
    return inputs, ilen, langs

def rank_reference_paragraphs(text_ranked_by, references_content, normalize=True, paragraph_ranking_method="TFIDF-title", LM=None, params=None, dictionary=None):
  
  if paragraph_ranking_method == "TFIDF-title":
    """ Rank and return reference paragraphs by tf-idf score on title tokens. """
    normalized_title = _normalize_text(text_ranked_by)
    title_tokens = _tokens_to_score(
        set(tokenizer.encode(text_encoder.native_to_unicode(normalized_title))))
    ref_paragraph_info = []
    doc_counts = collections.defaultdict(int)
    for ref in references_content:
      for paragraph in ref.split("\n"):
        normalized_paragraph = _normalize_text(paragraph)
        if cc_utils.filter_paragraph(normalized_paragraph):
          # Skip paragraph
          continue
        counts = _token_counts(normalized_paragraph, title_tokens)
        for token in title_tokens:
          if counts[token]:
            doc_counts[token] += 1
        content = normalized_paragraph if normalize else paragraph
        info = {"content": content, "counts": counts}
        ref_paragraph_info.append(info)

    for info in ref_paragraph_info:
      score = 0.
      for token in title_tokens:
        term_frequency = info["counts"][token]
        inv_doc_frequency = (
            float(len(ref_paragraph_info)) / max(doc_counts[token], 1))
        score += term_frequency * math.log(inv_doc_frequency)
      info["score"] = score

    ref_paragraph_info.sort(key=lambda el: el["score"], reverse=True)
    return [info["content"] for info in ref_paragraph_info]
  elif paragraph_ranking_method == "TFIDF-paragraph":
    """ Rank and return reference paragraphs by tf-idf score on article tokens. """
    # 統計references的token
    ref_paragraph_info = []
    doc_counts = collections.defaultdict(int)
    for ref in references_content:
      for paragraph in ref.split("\n"):
        normalized_paragraph = _normalize_text(paragraph)
        if cc_utils.filter_paragraph(normalized_paragraph):
          # Skip paragraph
          continue

        paragraph_tokens = _tokens_to_score(set(tokenizer.encode(text_encoder.native_to_unicode(normalized_paragraph))))
        counts = _token_counts(normalized_paragraph, paragraph_tokens)
        for token in paragraph_tokens:
          if counts[token]:
            doc_counts[token] += 1

        content = normalized_paragraph if normalize else paragraph
        info = {"content": content, "counts": counts}
        ref_paragraph_info.append(info)
        
    # 統計article的token
    normalized_article = _normalize_text(text_ranked_by)
    article_tokens = _tokens_to_score(
        set(tokenizer.encode(text_encoder.native_to_unicode(normalized_article))))
    article_counts = _token_counts(normalized_article, article_tokens)
    for token in article_tokens:
      if article_counts[token]:
        doc_counts[token] += 1
        
    # 計算article的TFIDF向量
    article_vec = []
    n_doc = len(ref_paragraph_info) + 1
    for token in doc_counts:
      if token in article_counts:
        term_frequency = article_counts[token]
        inv_doc_frequency = float(n_doc) / max(doc_counts[token], 1)
        article_vec.append(term_frequency * math.log(inv_doc_frequency))
      else:
        article_vec.append(0)
      
    article_vec = np.array(article_vec)
    
    # 計算references的TFIDF向量
    # 計算references和article的相似度
    for info in ref_paragraph_info:
      vec = []
      for token in doc_counts:
        if token in info["counts"]:
          term_frequency = info["counts"][token]
          inv_doc_frequency = float(n_doc) / max(doc_counts[token], 1)
          vec.append(term_frequency * math.log(inv_doc_frequency))
        else:
          vec.append(0)
      
      vec = np.array(vec)
      cos = np.dot(article_vec, vec) / (la.norm(article_vec) * la.norm(vec))  # 餘弦相似
      info["score"] = cos
    
    ref_paragraph_info.sort(key=lambda el: el["score"], reverse=True)
    return [info["content"] for info in ref_paragraph_info]
  else:
    """
    Rank and return reference paragraphs by tf-idf score on article tokens.
    text_ranked_by: [np.array(int32)]
    references_content: [np.array(int32)]
    """
    dataset = Ranking_Dataset(text_ranked_by, references_content, maxlen=params.bptt, dico=dictionary)
    dataloader=DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate(samples, params.pad_index))
        
    sent_embs = []
    for batch in dataloader:
      inputs, ilen, langs = [t.cuda() for t in batch]
      output = LM('fwd', x=inputs, lengths=ilen, langs=langs, causal=False)    # (slen, bs, dim)
      mean = output.sum(dim=0) / ilen.unsqueeze(-1)    # (bs, dim)
      sent_embs.append(mean)

    # Compute the cosine similarities.
    sent_embs = torch.cat(sent_embs, dim=0)
    zh_intro_emb = sent_embs[-1:]    # (1, dim)
    ref_embs = sent_embs[:-1]    # (#ref, dim)
    cos_sims = F.cosine_similarity(ref_embs, zh_intro_emb).tolist()
    assert len(references_content) == len(cos_sims)

    ref_paragraph_info = list(zip(references_content, cos_sims))
    ref_paragraph_info.sort(key=lambda el: el[1], reverse=True)
    return [info[0] for info in ref_paragraph_info]

def produce_examples(shard_ids, wikis_dir, refs_dir, urls_dir, vocab_path,
                     out_filepaths):
  """Produce examples from shard_ids to out_filepaths."""
  # * Join the Wikipedia articles with their references
  # * Run Tf-idf to sort reference paragraphs
  # * Encode the Wikipedia and reference text with the vocabulary
  # * Write out TFRecords of tensorflow.Example
  tf.logging.info("Processing %d input shards into %d output files.",
                  len(shard_ids), len(out_filepaths))

  vocab = text_encoder.SubwordTextEncoder(vocab_path)
  eot_ids = vocab.encode(EOT)

  def example_generator():
    """Generate Example dicts."""
    stats = dict(total_original_wikis=0, total_original_refs=0,
                 total_found_refs=0, ref_lengths=[], wiki_original_refs=[],
                 wiki_found_refs=[], wikis_skipped_no_refs=0,
                 wikis_skipped_short_lead=0, num_wikis_written=0)
    ref_files_by_shard = _references_files_by_shard(refs_dir)
    for shard_id in shard_ids:
      tf.logging.info("Processing shard %d", shard_id)
      wiki_urls = _wiki_urls_for_shard(shard_id, urls_dir)
      tf.logging.info("Loaded wiki URLs for shard")
      refs_content = _references_content(ref_files_by_shard[shard_id])
      tf.logging.info("Loaded reference content for shard")
      for i, wiki in enumerate(_wiki_articles(shard_id, wikis_dir)):
        if not i % 1000:
          tf.logging.info("Processing wiki index %d for shard %d", i, shard_id)
        stats["total_original_wikis"] += 1

        # Get reference content
        wiki_ref_content = []
        ref_urls = wiki_urls[wiki.url]["refs"]
        stats["total_original_refs"] += len(ref_urls)
        stats_wiki_original_refs = len(ref_urls)
        stats_wiki_found_refs = 0
        for ref_url in ref_urls:
          ref_content = refs_content.get(ref_url)
          if not ref_content:
            continue
          stats["total_found_refs"] += 1
          stats["ref_lengths"].append(len(ref_content))
          stats_wiki_found_refs += 1
          wiki_ref_content.append(ref_content)

        stats["wiki_original_refs"].append(stats_wiki_original_refs)
        stats["wiki_found_refs"].append(stats_wiki_found_refs)
        if not wiki_ref_content or len(wiki_ref_content) < _MIN_REFS:
          # No/few refs were found
          stats["wikis_skipped_no_refs"] += 1
          continue

        # Rank reference paragraphs with TFIDF
        wiki_title = _normalize_text(wiki.title)
        ranked_paragraphs = rank_reference_paragraphs(wiki_title,
                                                      wiki_ref_content)

        # Construct inputs from Wiki title and references
        inputs = []
        inputs.extend(vocab.encode(wiki_title))
        inputs.extend(eot_ids)
        for paragraph in ranked_paragraphs:
          if len(inputs) >= 1e6:
            break
          paragraph += " "
          inputs.extend(vocab.encode(paragraph))

        # Construct targets from article sections
        targets, section_boundaries = _encode_wiki_sections(
            wiki.sections, vocab)

        # Skip if lead section is too short
        if (not section_boundaries or
            section_boundaries[0] < _MIN_LEADSECTION_TOKENS):
          stats["wikis_skipped_short_lead"] += 1
          continue

        inputs.append(text_encoder.EOS_ID)
        targets.append(text_encoder.EOS_ID)

        stats["num_wikis_written"] += 1
        yield {
            "inputs": inputs,
            "targets": targets,
            "section_boundaries": section_boundaries,
        }

    tf.logging.info("Total: %d, Skipped: %d",
                    stats["num_wikis_written"],
                    stats["total_original_wikis"] - stats["num_wikis_written"])
    tf.logging.info("Total refs: %d, Skipped refs: %d",
                    stats["total_found_refs"],
                    stats["total_original_refs"] - stats["total_found_refs"])
    stats_fname = os.path.join(os.path.split(out_filepaths[0])[0],
                               "stats.%d.json" % shard_ids[0])
    with tf.gfile.Open(stats_fname, "w") as f:
      f.write(json.dumps(stats))

  generator_utils.generate_files(example_generator(), out_filepaths)

def produce_dataset(shard_ids, wikis_dir, refs_dir, urls_dir, out_dir, from_WikiAPI, paragraph_ranking_method, PID=0, LM=None, params=None):
  # * Join the Wikipedia articles with their references
  # * Run Tf-idf to sort reference paragraphs
  # * Encode the Wikipedia and reference text with the vocabulary
  # * Write out TFRecords of tensorflow.Example
  start_time=time.time()
  tf.logging.info("Process %d: Processing %d input shards.", PID, len(shard_ids))
  vocab_path="/home/zchen/XLM/data/processed/XLM_en_zh/50k_server153/vocab"
  dictionary = Dictionary.read_vocab(vocab_path)

  #本可用傳參的方式傳入bpe，但由於要執行multiprocessing，無法用pickle序列化，所以定義於此。
  codes_path="/home/zchen/XLM/data/processed/XLM_en_zh/50k_server153/codes"
  bpe = fastBPE.fastBPE(codes_path, vocab_path)

  stats = dict(start=shard_ids[0],
                total_original_wikis=0, total_original_refs=0,
                total_found_refs=0, ref_lengths=[], wiki_original_refs=[],
                wiki_found_refs=[], wikis_skipped_no_refs=0,
                wikis_skipped_short_en_intro=0, wikis_skipped_short_zh_intro=0, num_wikis_written=0)
  start=stats["start"]
  ref_files_by_shard = _references_files_by_shard(refs_dir)

  # 載入非訓練資料
  file_path = "/home/zchen/XLM/wikisum/en2zh_dataset/non_training_articles.json"
  with open(file_path, 'r', encoding='utf-8') as f:
      non_training_titles = set(json.load(f))

  def tokenize(text, lg):
    command=f'tools/tokenize.sh {lg} 2>&-'
    CompletedProcess=subprocess.run(command, input=text, cwd="/home/zchen/XLM/", encoding="utf-8", shell=True, stdout=subprocess.PIPE)

    return CompletedProcess.stdout[:-1]  # tokenize.sh會多加一個'\n'在最後

  def preprocess(text, lg):
    tick=time.time()
    t=tokenize(text, lg)
    elapsed=time.time()-tick
    print(f"Process {PID}: tokenize用時： {elapsed//3600:.0f}小時{elapsed//60%60:.0f}分{elapsed%60:.6f}秒")
    
    tick=time.time()
    postBPE_data=bpe.apply(t.split('\n'))
    elapsed=time.time()-tick
    print(f"Process {PID}: bpe用時： {elapsed//3600:.0f}小時{elapsed//60%60:.0f}分{elapsed%60:.6f}秒")
    
    return postBPE_data

  def concat(sentences):
    sentences=[sent.lower() for sent in sentences]
    
    return "\nS\n".join(sentences)

  def data_extractor(sentences):
    offs=0
    last=len(sentences)-1
    for i, sentence in enumerate(sentences):
      if sentence == 'S':
        yield sentences[offs:i]

        offs=i+1
      elif i == last:
        yield sentences[offs:]

  def data_generator(DATA):
    en_title_extractor=data_extractor(DATA["inputs"]["title"])
    paragraphs_extractor=data_extractor(DATA["inputs"]["paragraphs"])

    zh_title_extractor=data_extractor(DATA["targets"]["title"])
    intro_extractor=data_extractor(DATA["targets"]["intro"])

    for en_title, paragraphs, zh_title, intro in zip(en_title_extractor, paragraphs_extractor, zh_title_extractor, intro_extractor):
      # Construct inputs from Wiki title and references
      inputs = {}
      title=Dictionary.index_sentences(en_title, dictionary)
      if len(title["sentences"]) != 1:
        file=out_dir+"log/debug"
        with tf.gfile.Open(file, 'a') as f:
          f.write(f"process {PID}:\n")
          f.write(f"shard {shard_id}:\n")
          f.write(f"input title: {title['sentences']}\n\n")
        yield None
        continue
      title["sentence"]=title["sentences"][0]
      del title["sentences"]
      inputs["title"]=title
      inputs["paragraphs"]=Dictionary.index_sentences(paragraphs, dictionary)
        # dict:
        #     {
        #       'sentences': [np.array(int32)],
        #       'unk_words': dict,
        #     }

      # Construct targets from article sections
      targets={}
      title=Dictionary.index_sentences(zh_title, dictionary)
      if len(title["sentences"]) != 1:
        file=out_dir+"log/debug"
        with tf.gfile.Open(file, 'a') as f:
          f.write(f"process {PID}:\n")
          f.write(f"shard {shard_id}:\n")
          f.write(f"target title: {title['sentences']}\n\n")
          yield None
          continue
      title["sentence"]=title["sentences"][0]
      del title["sentences"]
      targets["title"]=title
      targets["intro"]=Dictionary.index_sentences(intro, dictionary)

      yield {
          "inputs": inputs,
          "targets": targets
          }

  #中斷點
  log=out_dir+f"log/log{PID}.json"
  if tf.gfile.Exists(log):
      with tf.gfile.Open(log, 'r') as f:
          stats=json.load(f)
          start=stats["start"]+1
      
      print(f"Process {PID}: Start from shard {start}......")
      
  for shard_id in shard_ids:
    if shard_id < start: continue
    
    print(f"Process {PID}: Processing shard {shard_id}")
    wiki_urls = _wiki_urls_for_shard(shard_id, urls_dir)
    tf.logging.info("Process %d: Loaded wiki URLs for shard", PID)
    refs_content = {text_encoder.to_unicode(k): v for k, v in _references_content(ref_files_by_shard[shard_id]).items()}
    tf.logging.info("Process %d: Loaded reference content for shard", PID)
    
    zh_articles={}
    DATA={
      "inputs": {"title": [], "paragraphs": []},
      "targets": {"title": [], "intro": []}
      }
    for i, (en_wiki, zh_title, zh_intro) in enumerate(generate_wiki_data(shard_id, wikis_dir)):
      print(PID)

      if paragraph_ranking_method == "TFIDF-paragraph":
        en_intro = en_wiki.sections[0].text
        
        # Skip if intro is too short
        if (len(en_intro) < _MIN_LEADSECTION_TOKENS):
          stats["wikis_skipped_short_en_intro"] += 1
          continue

      if (len(zh_intro) < _MIN_LEADSECTION_TOKENS):
        stats["wikis_skipped_short_zh_intro"] += 1
        continue
      
      stats["total_original_wikis"] += 1

      # Get reference content
      wiki_ref_content = []
      ref_urls = wiki_urls[en_wiki.url]["refs"]
      stats["total_original_refs"] += len(ref_urls)
      stats_wiki_original_refs = len(ref_urls)
      stats_wiki_found_refs = 0
      for ref_url in ref_urls:
        ref_content = refs_content.get(ref_url)
        if not ref_content:
          continue
        stats["total_found_refs"] += 1
        stats["ref_lengths"].append(len(ref_content))
        stats_wiki_found_refs += 1
        wiki_ref_content.append(ref_content)
      
      stats["wiki_original_refs"].append(stats_wiki_original_refs)
      stats["wiki_found_refs"].append(stats_wiki_found_refs)
      if not wiki_ref_content or len(wiki_ref_content) < _MIN_REFS:
        # No/few refs were found
        stats["wikis_skipped_no_refs"] += 1
        continue

      wiki_title = _normalize_text(en_wiki.title)

      if paragraph_ranking_method != "LM":
        """ Rank reference """
        if paragraph_ranking_method == "TFIDF-title":
          text_ranked_by = wiki_title
        elif paragraph_ranking_method == "TFIDF-paragraph":
          text_ranked_by = _normalize_text(en_intro)
        ranked_paragraphs = rank_reference_paragraphs(text_ranked_by, wiki_ref_content, paragraph_ranking_method)
        if not ranked_paragraphs: continue  # rank_reference_paragraphs()中會做篩選
        
        paragraphs=[]
        n_tokens=0
        for paragraph in ranked_paragraphs:
          n_tokens+=len(paragraph)
          if n_tokens >= 256**2:
            break
          paragraphs.append(paragraph)

        if not paragraphs: continue  # 若第一段太長，直接捨棄。
      else:
        paragraphs=[]
        for ref in wiki_ref_content:
          for paragraph in ref.split("\n"):
            normalized_paragraph = _normalize_text(paragraph)
            if cc_utils.filter_paragraph(normalized_paragraph):
              # Skip paragraph
              continue
            paragraphs.append(normalized_paragraph)
      paragraphs='\n'.join(paragraphs)
      
      # 儲存中文條目
      if zh_title in zh_articles: continue
      zh_articles[zh_title] = zh_intro

      # 由於一筆一筆做前處理效率太低，先蒐集好這個shard的所有資料，一起做前處理。
      # Construct inputs from Wiki title and references
      DATA["inputs"]["title"].append(wiki_title)
      DATA["inputs"]["paragraphs"].append(paragraphs)

      # Construct targets from article sections
      DATA["targets"]["title"].append(zh_title)
      DATA["targets"]["intro"].append(zh_intro)
    
    # 這個shard中所有資料一起做前處理
    text=concat(DATA["inputs"]["title"])
    DATA["inputs"]["title"]=preprocess(text, "en")

    text=concat(DATA["inputs"]["paragraphs"])
    DATA["inputs"]["paragraphs"]=preprocess(text, "en")

    text=concat(DATA["targets"]["title"])
    DATA["targets"]["title"]=preprocess(text, "zh")

    text=concat(DATA["targets"]["intro"])
    DATA["targets"]["intro"]=preprocess(text, "zh")

    # 產生資料集
    dataset=[]
    index = {
      "training": [],
      "non_training": []
    }
    zh_titles = list(zh_articles.keys())
    pos, length=0, 0
    for i, data in enumerate(data_generator(DATA)):
      if data == None:
        del zh_articles[zh_titles[i]]
        continue

      if paragraph_ranking_method == "LM":
        """ Rank reference """
        ranked_paragraphs = rank_reference_paragraphs(
          text_ranked_by=data["targets"]["intro"]["sentences"],
          references_content=data["inputs"]["paragraphs"]["sentences"],
          paragraph_ranking_method=paragraph_ranking_method,
          LM=LM, params=params, dictionary=dictionary)
        if not ranked_paragraphs:
          del zh_articles[zh_titles[i]]
          continue  # rank_reference_paragraphs()中會做篩選
        
        paragraphs=[]
        n_tokens=0
        for paragraph in ranked_paragraphs:
          n_tokens+=len(paragraph)
          if n_tokens >= 256**2:
            break
          paragraphs.append(paragraph)

        if not paragraphs:
          del zh_articles[zh_titles[i]]
          continue  # 若第一段太長，直接捨棄。

        data["inputs"]["paragraphs"]["sentences"] = paragraphs

      pkl_data=pickle.dumps(data)
      dataset.append(pkl_data)

      pos+=length
      length=len(pkl_data)
      if zh_titles[i] in non_training_titles:
        index["non_training"].append([pos, length])  # [position, length]
      else:
        index["training"].append([pos, length])  # [position, length]
      
      stats["num_wikis_written"] += 1
      
    if len(dataset) != len(zh_articles):
      file=out_dir+"log/debug"
      with tf.gfile.Open(file, 'a') as f:
        f.write(f"process {PID}:\n")
        f.write(f"shard {shard_id}: {len(dataset)} {len(zh_articles)}\n\n")

    if from_WikiAPI:
      file = "/home/zchen/XLM/wikisum/en2zh_dataset/zh_articles/zh_articles-{shard_id:0>5d}-of-01000.json"
      with tf.gfile.Open(file, "w") as f:
        json.dump(zh_articles, f, ensure_ascii=False)

    file=out_dir+f"dataset/dataset-{shard_id:0>5d}-of-01000.pkl"
    with tf.gfile.Open(file, "wb") as f:
      for data in dataset:
        f.write(data)

    file=out_dir+f"dataset/index-{shard_id:0>5d}-of-01000.json"
    with tf.gfile.Open(file, "w") as f:
      json.dump(index, f, ensure_ascii=False)

    # 儲存中斷點
    stats["start"]=shard_id
    with tf.gfile.Open(log, 'w') as f:
      json.dump(stats, f, ensure_ascii=False)

    elapsed=time.time()-start_time
    print(f"Process {PID}: 進度： {shard_id-shard_ids[0]+1}/{len(shard_ids)} | 用時： {elapsed//3600:.0f}小時{elapsed//60%60:.0f}分{elapsed%60:.0f}秒")
    print()
    
  tf.logging.info("Process %d: Total: %d, Skipped: %d", PID,
                  stats["num_wikis_written"],
                  stats["total_original_wikis"] - stats["num_wikis_written"])
  tf.logging.info("Process %d: Total refs: %d, Skipped refs: %d", PID,
                  stats["total_found_refs"],
                  stats["total_original_refs"] - stats["total_found_refs"])
                  
# debug用
def preprocess(text, lg, bpe):
  voc_path="/home/zchen/XLM/data/processed/XLM_en_zh/50k_server153/vocab"
  dictionary = Dictionary.read_vocab(voc_path)

  command=f'tools/tokenize.sh {lg} 2>&-'
  t=subprocess.run(command, input=text, cwd="/home/zchen/XLM/", encoding="utf-8", shell=True, stdout=subprocess.PIPE).stdout[:-1]  # tokenize.sh會多加一個'\n'在最後
  print(t)

  postBPE_data=bpe.apply(t.split('\n'))
  print(postBPE_data)

  t=Dictionary.index_sentences(postBPE_data, dictionary)
  print(t)

def _format_title(title):
  return " == %s == " % title


def _encode_wiki_sections(sections, vocab):
  """Encodes sections with vocab. Returns ids and section boundaries."""
  ids = []
  section_boundaries = []
  for i, section in enumerate(sections):
    if i > 0:
      # Skip including article title
      ids.extend(vocab.encode(_format_title(_normalize_text(section.title))))
    ids.extend(vocab.encode(_normalize_text(section.text)))
    section_boundaries.append(len(ids))

  return ids, section_boundaries


def _process_folders(tmp_dir):
  return tf.gfile.Glob(os.path.join(tmp_dir, PROCESS_FOLDER_PREFIX) + "*")

def referenced_by_linked_articles(ref_url, shard_id):
  #�P�_�b��wiki_urls�ɮפ��A�O�_��linked_article�ѦҨ즹�ѦҤ��m
  #�^�ǥ��L��
  linked_articles_file=LINKED_ARTICLES_FILE.format(shard_id)
  with tf.gfile.Open(linked_articles_file) as f:
      linked_articles = json.load(f)
      
  wiki_urls_file=os.path.join(WIKI_URLS_DIR, WIKI_URLS_FILE%shard_id)
  with tf.gfile.Open(wiki_urls_file) as f:
      wiki_urls = json.load(f)
      
  #�M���r�媺�C��key(linked_article)
  for art in linked_articles:
    if ref_url in wiki_urls.get(art): return True
    
  return False

def extract_references_from_wets(wet_files, metadata_dir, out_dir,
                                 tmp_dir=None):
  """Extract references from WET files into sharded output files."""
  # Setup output files
  shard_files = make_ref_shard_files(out_dir)

  num_refs = 0
  for i, wet_file in enumerate(wet_files):
    num_refs_in_wet = 0
    tf.logging.info("Processing file %d", i)

    # Read metadata file
    metadata_fname = os.path.join(
        metadata_dir, os.path.basename(wet_file)) + cc_utils.METADTA_SUFFIX
    with tf.gfile.Open(cc_utils.readahead(metadata_fname)) as f:
      wet_metadata = json.loads(f.read())

    if not wet_metadata:
      # No references in this WET file
      continue

    if wet_file.startswith("http"):
      # download
      if not tmp_dir:
        tmp_dir = tempfile.gettempdir()
      record_gen = cc_utils.wet_records_from_url(wet_file, tmp_dir)
    else:
      # local
      record_gen = cc_utils.wet_records_from_file_obj(
          cc_utils.gzip_memfile(wet_file), take_ownership=True)

    for wet_record in record_gen:
      shard_ids = wet_metadata.get(wet_record.url)
      #�qshard_ids���z��X�t��linked_article��id
      shard_ids=[shard_id for shard_id in shard_ids if referenced_by_linked_articles(wet_record.url, shard_id)]
      if not shard_ids:
        # URL not in dataset
        continue

      # Serialize and write out
      ex = _make_example_from_record(wet_record)
      ex_str = ex.SerializeToString()
      for shard_id in shard_ids:
        shard_files[shard_id].write(ex_str)
      num_refs += 1
      num_refs_in_wet += 1

    tf.logging.info("Wrote out %d references for this WET", num_refs_in_wet)

  tf.logging.info("Wrote out %d references total", num_refs)

  # Cleanup
  for shard_file in shard_files:
    shard_file.close()
