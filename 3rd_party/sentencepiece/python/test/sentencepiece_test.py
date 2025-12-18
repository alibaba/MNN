#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2018 Google Inc.
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
# limitations under the License.!

from collections import defaultdict
import io
import os
import pickle
import sys
import threading
import unittest
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
print(HERE)

print('VERSION={}'.format(spm.__version__))

data_dir = HERE


class TestSentencepieceProcessor(unittest.TestCase):
  """Test case for SentencePieceProcessor"""

  def setUp(self):
    self.sp_ = spm.SentencePieceProcessor()
    self.jasp_ = spm.SentencePieceProcessor()
    self.assertTrue(self.sp_.Load(os.path.join(HERE, 'test_model.model')))
    self.assertTrue(self.jasp_.Load(os.path.join(HERE, 'test_ja_model.model')))
    with open(os.path.join(HERE, 'test_model.model'), 'rb') as f:
      self.assertTrue(self.sp_.LoadFromSerializedProto(f.read()))
    with open(os.path.join(HERE, 'test_ja_model.model'), 'rb') as f:
      self.assertTrue(self.jasp_.LoadFromSerializedProto(f.read()))

  def test_load(self):
    self.assertEqual(1000, self.sp_.GetPieceSize())
    self.assertEqual(0, self.sp_.PieceToId('<unk>'))
    self.assertEqual(1, self.sp_.PieceToId('<s>'))
    self.assertEqual(2, self.sp_.PieceToId('</s>'))
    self.assertEqual('<unk>', self.sp_.IdToPiece(0))
    self.assertEqual('<s>', self.sp_.IdToPiece(1))
    self.assertEqual('</s>', self.sp_.IdToPiece(2))
    self.assertEqual(0, self.sp_.unk_id())
    self.assertEqual(1, self.sp_.bos_id())
    self.assertEqual(2, self.sp_.eos_id())
    self.assertEqual(-1, self.sp_.pad_id())
    for i in range(self.sp_.GetPieceSize()):
      piece = self.sp_.IdToPiece(i)
      self.assertEqual(i, self.sp_.PieceToId(piece))

    self.assertEqual(1000, self.sp_.get_piece_size())
    self.assertEqual(0, self.sp_.piece_to_id('<unk>'))
    self.assertEqual(1, self.sp_.piece_to_id('<s>'))
    self.assertEqual(2, self.sp_.piece_to_id('</s>'))
    self.assertEqual('<unk>', self.sp_.id_to_piece(0))
    self.assertEqual('<s>', self.sp_.id_to_piece(1))
    self.assertEqual('</s>', self.sp_.id_to_piece(2))
    for i in range(self.sp_.get_piece_size()):
      piece = self.sp_.id_to_piece(i)
      self.assertEqual(i, self.sp_.piece_to_id(piece))

  def test_roundtrip(self):
    text = 'I saw a girl with a telescope.'
    ids = self.sp_.EncodeAsIds(text)
    pieces1 = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.NBestEncodeAsPieces(text, 10)[0]
    self.assertEqual(pieces1, pieces2)
    self.assertEqual(text, self.sp_.DecodePieces(pieces1))
    self.assertEqual(text, self.sp_.DecodeIds(ids))
    for n in range(100):
      self.assertEqual(
          text,
          self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, 64, 0.5)),
      )
      self.assertEqual(
          text,
          self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, -1, 0.5)),
      )
      self.assertEqual(
          text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, 64, 0.5))
      )
      self.assertEqual(
          text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, -1, 0.5))
      )

    ids2 = self.sp_.encode_as_ids(text)
    pieces3 = self.sp_.encode_as_pieces(text)
    pieces4 = self.sp_.nbest_encode_as_pieces(text, 10)[0]
    self.assertEqual(pieces3, pieces4)
    self.assertEqual(pieces1, pieces3)
    self.assertEqual(ids, ids2)
    self.assertEqual(text, self.sp_.decode_pieces(pieces3))
    self.assertEqual(text, self.sp_.decode_ids(ids2))
    for n in range(100):
      self.assertEqual(
          text,
          self.sp_.decode_pieces(
              self.sp_.sample_encode_as_pieces(text, 64, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.sp_.decode_pieces(
              self.sp_.sample_encode_as_pieces(text, -1, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, 64, 0.5)),
      )
      self.assertEqual(
          text,
          self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, -1, 0.5)),
      )

    self.assertEqual(
        self.sp_.calculate_entropy(text, 0.1),
        self.sp_.CalculateEntropy(text, 0.1),
    )

  def test_ja_load(self):
    self.assertEqual(8000, self.jasp_.GetPieceSize())
    self.assertEqual(0, self.jasp_.PieceToId('<unk>'))
    self.assertEqual(1, self.jasp_.PieceToId('<s>'))
    self.assertEqual(2, self.jasp_.PieceToId('</s>'))
    self.assertEqual('<unk>', self.jasp_.IdToPiece(0))
    self.assertEqual('<s>', self.jasp_.IdToPiece(1))
    self.assertEqual('</s>', self.jasp_.IdToPiece(2))
    for i in range(self.jasp_.GetPieceSize()):
      piece = self.jasp_.IdToPiece(i)
      self.assertEqual(i, self.jasp_.PieceToId(piece))

    self.assertEqual(8000, self.jasp_.get_piece_size())
    self.assertEqual(0, self.jasp_.piece_to_id('<unk>'))
    self.assertEqual(1, self.jasp_.piece_to_id('<s>'))
    self.assertEqual(2, self.jasp_.piece_to_id('</s>'))
    self.assertEqual('<unk>', self.jasp_.id_to_piece(0))
    self.assertEqual('<s>', self.jasp_.id_to_piece(1))
    self.assertEqual('</s>', self.jasp_.id_to_piece(2))
    for i in range(self.jasp_.get_piece_size()):
      piece = self.jasp_.id_to_piece(i)
      self.assertEqual(i, self.jasp_.piece_to_id(piece))

  def test_ja_roundtrip(self):
    text = '清水寺は京都にある。'
    ids = self.jasp_.EncodeAsIds(text)
    pieces1 = self.jasp_.EncodeAsPieces(text)
    pieces2 = self.jasp_.NBestEncodeAsPieces(text, 10)[0]
    self.assertEqual(pieces1, pieces2)
    self.assertEqual(text, self.jasp_.DecodePieces(pieces1))
    self.assertEqual(text, self.jasp_.DecodeIds(ids))
    for n in range(100):
      self.assertEqual(
          text,
          self.jasp_.DecodePieces(
              self.jasp_.SampleEncodeAsPieces(text, 64, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.jasp_.DecodePieces(
              self.jasp_.SampleEncodeAsPieces(text, -1, 0.5)
          ),
      )

    ids2 = self.jasp_.encode_as_ids(text)
    pieces3 = self.jasp_.encode_as_pieces(text)
    pieces4 = self.jasp_.nbest_encode_as_pieces(text, 10)[0]
    self.assertEqual(pieces3, pieces4)
    self.assertEqual(pieces1, pieces3)
    self.assertEqual(ids, ids2)
    self.assertEqual(text, self.jasp_.decode_pieces(pieces1))
    self.assertEqual(text, self.jasp_.decode_ids(ids2))
    for n in range(100):
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, 64, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, -1, 0.5)
          ),
      )

      self.assertEqual(
          self.jasp_.calculate_entropy(text, 0.1),
          self.jasp_.CalculateEntropy(text, 0.1),
      )

  def test_train(self):
    tid = threading.get_native_id()
    spm.SentencePieceTrainer.Train(
        '--input='
        + os.path.join(data_dir, 'botchan.txt')
        + f' --model_prefix=m_{tid} --vocab_size=1000'
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(f'm_{tid}.model')
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

  def test_train_iterator(self):
    tid = threading.get_native_id()
    spm.SentencePieceTrainer.Train(
        '--input='
        + os.path.join(data_dir, 'botchan.txt')
        + f' --model_prefix=m_{tid} --vocab_size=1000'
    )
    # Load as 'rb' for Python3.5/2.7.
    os1 = io.BytesIO()
    os2 = io.BytesIO()

    # suppress logging (redirect to /dev/null)
    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix=f'm_{tid}',
        vocab_size=1000,
        # logstream=open(os.devnull, 'w'),
    )

    with open(os.path.join(data_dir, 'botchan.txt'), 'rb') as is1:
      spm.SentencePieceTrainer.train(
          sentence_iterator=is1,
          model_prefix=f'm_{tid}',
          vocab_size=1000,
          # logstream=open(os.devnull, 'w'),
      )

    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_writer=os1,
        vocab_size=1000,
        # logstream=open(os.devnull, 'w'),
    )

    with open(os.path.join(data_dir, 'botchan.txt'), 'rb') as is2:
      spm.SentencePieceTrainer.train(
          sentence_iterator=is2,
          model_writer=os2,
          vocab_size=1000,
          # logstream=open(os.devnull, 'w'),
      )

    sp1 = spm.SentencePieceProcessor(model_proto=os1.getvalue())
    sp2 = spm.SentencePieceProcessor(model_proto=os2.getvalue())
    self.assertEqual(
        [sp1.id_to_piece(i) for i in range(sp1.get_piece_size())],
        [sp2.id_to_piece(i) for i in range(sp2.get_piece_size())],
    )

  def test_train_kwargs(self):
    tid = threading.get_native_id()
    # suppress logging (redirect to /dev/null)
    spm.SentencePieceTrainer.train(
        input=[os.path.join(data_dir, 'botchan.txt')],
        model_prefix=f'm_{tid}',
        vocab_size=1002,
        user_defined_symbols=['foo', 'bar', ',', ' ', '\t', '\b', '\n', '\r'],
        # logstream=open(os.devnull, 'w'),
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(f'm_{tid}.model')
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

    s = 'hello\tworld\r\nthis\tis a \b pen'
    self.assertEqual(s, sp.decode(sp.encode(s)))

  def test_serialized_proto(self):
    text = 'I saw a girl with a telescope.'
    s1 = self.sp_.EncodeAsSerializedProto(text)
    s2 = self.sp_.SampleEncodeAsSerializedProto(text, 10, 0.2)
    s3 = self.sp_.NBestEncodeAsSerializedProto(text, 10)
    s4 = self.sp_.DecodePiecesAsSerializedProto(['foo', 'bar'])
    s5 = self.sp_.DecodeIdsAsSerializedProto([20, 30])

    t1 = self.sp_.encode_as_serialized_proto(text)
    t2 = self.sp_.sample_encode_as_serialized_proto(text, 10, 0.2)
    t3 = self.sp_.nbest_encode_as_serialized_proto(text, 10)
    t4 = self.sp_.decode_pieces_as_serialized_proto(['foo', 'bar'])
    t5 = self.sp_.decode_ids_as_serialized_proto([20, 30])

    y1 = self.sp_.encode(text, out_type='serialized_proto')
    y2 = self.sp_.encode(
        text, enable_sampling=True, out_type='serialized_proto'
    )
    y3 = self.sp_.nbest_encode(text, out_type='serialized_proto', nbest_size=10)
    y4 = self.sp_.decode(['foo', 'bar'], out_type='serialized_proto')
    y5 = self.sp_.decode([20, 30], out_type='serialized_proto')

    self.assertEqual(type(s1), bytes)
    self.assertEqual(type(s2), bytes)
    self.assertEqual(type(t2), bytes)
    self.assertEqual(type(s3), bytes)
    self.assertEqual(type(s4), bytes)
    self.assertEqual(type(s5), bytes)

    self.assertEqual(s1, t1)
    self.assertEqual(s3, t3)
    self.assertEqual(s4, t4)
    self.assertEqual(s5, t5)
    self.assertEqual(s1, y1)
    self.assertEqual(s3, y3)
    self.assertEqual(s4, y4)
    self.assertEqual(s5, y5)

    ids = self.jasp_.EncodeAsIds(text)
    pieces = self.jasp_.EncodeAsPieces(text)
    s1 = self.jasp_.EncodeAsSerializedProto(text)
    s2 = self.jasp_.DecodeIdsAsSerializedProto(ids)
    s3 = self.jasp_.DecodePiecesAsSerializedProto(ids)
    self.assertEqual(s2, s1)
    self.assertEqual(s3, s1)

  def test_decode_bytes(self):
    texts = ['Hello world', '清水寺は京都にある。']
    ids = self.jasp_.encode(texts, out_type=int)
    s1 = self.jasp_.decode(ids, out_type=bytes)
    s2 = self.jasp_.decode(ids, out_type=str)
    self.assertEqual(len(s1), 2)
    self.assertEqual(type(s1[0]), bytes)
    self.assertEqual(type(s1[1]), bytes)
    self.assertEqual(len(s2), 2)
    self.assertEqual(type(s2[0]), str)
    self.assertEqual(type(s2[1]), str)
    self.assertEqual(s1[0].decode(encoding='utf-8'), s2[0])
    self.assertEqual(s1[1].decode(encoding='utf-8'), s2[1])

    text = 'Hello world'
    ids = self.jasp_.encode(text, out_type=int)
    s1 = self.jasp_.decode(ids, out_type=bytes)
    s2 = self.jasp_.decode(ids, out_type=str)
    self.assertEqual(type(s1), bytes)
    self.assertEqual(type(s2), str)
    self.assertEqual(s1.decode(encoding='utf-8'), s2)

    x = self.jasp_.encode(text, out_type='immutable_proto')
    self.assertEqual(x.text, x.text_as_bytes.decode(encoding='utf-8'))
    for sp in x.pieces:
      self.assertEqual(sp.piece, sp.piece_as_bytes.decode(encoding='utf-8'))
      self.assertEqual(sp.surface, sp.surface_as_bytes.decode(encoding='utf-8'))

    x = self.jasp_.decode(ids, out_type='immutable_proto')
    self.assertEqual(x.text, x.text_as_bytes.decode(encoding='utf-8'))
    for sp in x.pieces:
      self.assertEqual(sp.piece, sp.piece_as_bytes.decode(encoding='utf-8'))
      self.assertEqual(sp.surface, sp.surface_as_bytes.decode(encoding='utf-8'))

  def test_immutable_proto(self):
    text = 'I saw a girl with a telescope.'
    s1 = self.sp_.EncodeAsImmutableProto(text)
    s2 = self.sp_.SampleEncodeAsImmutableProto(text, 10, 0.2)
    s3 = self.sp_.NBestEncodeAsImmutableProto(text, 10)
    s4 = self.sp_.DecodePiecesAsImmutableProto(['foo', 'bar'])
    s5 = self.sp_.DecodeIdsAsImmutableProto([20, 30])

    print(s1)
    print(s2)
    print(s3)
    print(s4)
    print(s5)

    t1 = self.sp_.encode_as_immutable_proto(text)
    t2 = self.sp_.sample_encode_as_immutable_proto(text, 10, 0.2)
    t3 = self.sp_.nbest_encode_as_immutable_proto(text, 10)
    t4 = self.sp_.decode_pieces_as_immutable_proto(['foo', 'bar'])
    t5 = self.sp_.decode_ids_as_immutable_proto([20, 30])

    y1 = self.sp_.encode(text, out_type='immutable_proto')
    y2 = self.sp_.encode(text, enable_sampling=True, out_type='immutable_proto')
    y3 = self.sp_.nbest_encode(text, out_type='immutable_proto', nbest_size=10)
    y4 = self.sp_.decode(['foo', 'bar'], out_type='immutable_proto')
    y5 = self.sp_.decode([20, 30], out_type='immutable_proto')

    self.assertEqual(s1, t1)
    self.assertEqual(s3, t3)
    self.assertEqual(s4, t4)
    self.assertEqual(s5, t5)
    self.assertEqual(s1, y1)
    self.assertEqual(s3, y3)
    self.assertEqual(s4, y4)
    self.assertEqual(s5, y5)

    hset_piece = defaultdict(int)

    # eq test
    for i in range(len(s1.pieces)):
      self.assertEqual(s1.pieces[i], t1.pieces[i])
      hset_piece[s1.pieces[i]] += 1
      hset_piece[t1.pieces[i]] += 1

    self.assertEqual(len(hset_piece), len(s1.pieces))

    # has test
    hset = defaultdict(int)
    hset[s1] += 1
    hset[t1] += 1
    hset[s3] += 1
    hset[t3] += 1

    self.assertEqual(len(hset), 2)
    self.assertEqual(hset[s1], 2)
    self.assertEqual(hset[s3], 2)
    self.assertEqual(hset[t1], 2)
    self.assertEqual(hset[t3], 2)

    x1 = self.sp_.encode_as_serialized_proto(text)
    x2 = self.sp_.sample_encode_as_serialized_proto(text, 10, 0.2)
    x3 = self.sp_.nbest_encode_as_serialized_proto(text, 10)
    x4 = self.sp_.decode_pieces_as_serialized_proto(['foo', 'bar'])
    x5 = self.sp_.decode_ids_as_serialized_proto([20, 30])

    self.assertEqual(x1, t1.SerializeAsString())
    self.assertEqual(x3, t3.SerializeAsString())
    self.assertEqual(x4, t4.SerializeAsString())
    self.assertEqual(x5, t5.SerializeAsString())

    v1 = self.sp_.EncodeAsIds(text)
    v2 = self.sp_.EncodeAsPieces(text)
    self.assertEqual([x.id for x in s1.pieces], v1)
    self.assertEqual([x.piece for x in s1.pieces], v2)
    self.assertEqual(text, s1.text)

    surfaces1 = [s1.text[x.begin : x.end] for x in s1.pieces]
    surfaces2 = [x.surface for x in s1.pieces]
    self.assertEqual(surfaces1, surfaces2)

    ids = []
    for i in range(len(s1.pieces)):
      ids.append(s1.pieces[i].id)
    self.assertEqual(ids, v1)

    pieces = []
    for i in range(len(s1.pieces)):
      pieces.append(s1.pieces[i].piece)
    self.assertEqual(pieces, v2)

    for v in s3.nbests:
      self.assertEqual(text, v.text)
      self.assertEqual(self.sp_.Decode([x.id for x in v.pieces]), text)

    for i in range(len(s3.nbests)):
      self.assertEqual(text, s3.nbests[i].text)
      self.assertEqual(
          self.sp_.Decode([x.id for x in s3.nbests[i].pieces]), text
      )

    # slice
    self.assertEqual(s1.pieces[::-1], list(reversed(s1.pieces)))
    self.assertEqual(s3.nbests[::-1], list(reversed(s3.nbests)))

    # Japanese offset
    s1 = self.jasp_.EncodeAsImmutableProto(
        '吾輩は猫である。Hello world. ABC 123'
    )
    surfaces1 = [s1.text[x.begin : x.end] for x in s1.pieces]
    surfaces2 = [x.surface for x in s1.pieces]
    self.assertEqual(surfaces1, surfaces2)

    ids = [x.id for x in s1.pieces]
    s2 = self.jasp_.DecodeIdsAsImmutableProto(ids)
    self.assertEqual(s2, s1)

    pieces = [x.piece for x in s1.pieces]
    s2 = self.jasp_.DecodePiecesAsImmutableProto(pieces)
    self.assertEqual(s2, s1)

  def test_new_api(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )
    text = 'hello world'
    text2 = 'Tokyo'
    ids = self.sp_.EncodeAsIds(text)
    ids2 = self.sp_.EncodeAsIds(text2)
    pieces = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.EncodeAsPieces(text2)
    sprotos = self.sp_.EncodeAsSerializedProto(text)
    sproto2 = self.sp_.EncodeAsSerializedProto(text2)
    iprotos = self.sp_.EncodeAsImmutableProto(text)
    iprotos2 = self.sp_.EncodeAsImmutableProto(text2)

    self.assertEqual(sp.encode(text, out_type=int), ids)
    self.assertEqual(sp.encode(text, out_type=str), pieces)
    self.assertEqual(sp.encode(text, out_type='serialized_proto'), sprotos)
    self.assertEqual(sp.encode(text, out_type='immutable_proto'), iprotos)

    self.assertEqual(sp.encode([text], out_type=int), [ids])
    self.assertEqual(sp.encode([text], out_type=str), [pieces])
    self.assertEqual(sp.encode([text], out_type='serialized_proto'), [sprotos])
    self.assertEqual(sp.encode([text], out_type='immutable_proto'), [iprotos])

    self.assertEqual(len(iprotos.pieces), len(pieces))
    self.assertEqual(len(iprotos.pieces), len(ids))
    self.assertEqual(iprotos.text, text)

    self.assertEqual(len(iprotos2.pieces), len(pieces2))
    self.assertEqual(len(iprotos2.pieces), len(ids2))
    self.assertEqual(iprotos2.text, text2)

    for i in range(len(iprotos.pieces)):
      self.assertEqual(ids[i], iprotos.pieces[i].id)
      self.assertEqual(pieces[i], iprotos.pieces[i].piece)

    for i, piece in enumerate(iprotos.pieces):
      self.assertEqual(ids[i], piece.id)
      self.assertEqual(pieces[i], piece.piece)

    for i in range(len(iprotos2.pieces)):
      self.assertEqual(ids2[i], iprotos2.pieces[i].id)
      self.assertEqual(pieces2[i], iprotos2.pieces[i].piece)

    for i, piece in enumerate(iprotos2.pieces):
      self.assertEqual(ids2[i], piece.id)
      self.assertEqual(pieces2[i], piece.piece)

    detok_ids = self.sp_.DecodeIds(ids)
    detok_pieces = self.sp_.DecodePieces(pieces)
    self.assertEqual(sp.decode(ids), detok_ids)
    self.assertEqual(sp.decode(pieces), detok_pieces)
    self.assertEqual(sp.decode([]), '')
    self.assertEqual(sp.decode([[]]), [''])

    # add_bos, add_eos, reverse
    self.assertEqual([sp.bos_id()] + ids, sp.encode(text, add_bos=True))
    self.assertEqual(ids + [sp.eos_id()], sp.encode(text, add_eos=True))
    self.assertEqual(ids + [sp.eos_id()], sp.EncodeAsIds(text, add_eos=True))
    rids = ids[:]
    rids.reverse()

    self.assertEqual(rids, sp.encode(text, reverse=True))
    self.assertEqual(rids, sp.EncodeAsIds(text, reverse=True))

    # different shape.
    self.assertEqual([ids, ids2], sp.encode([text, text2]))
    self.assertEqual([pieces, pieces2], sp.encode([text, text2], out_type=str))
    self.assertEqual([text, text2], sp.decode([ids, ids2]))
    self.assertEqual([text, text2], sp.decode([pieces, pieces2]))

    pieces = list(reversed(self.sp_.EncodeAsPieces(text)))
    self.assertEqual(pieces, sp.encode(text, reverse=True, out_type=str))

    # emit unk piece
    unk_char = '藤'
    pieces = self.sp_.EncodeAsIds(unk_char, emit_unk_piece=True)
    pieces2 = self.sp_.encode(unk_char, out_type=int, emit_unk_piece=True)
    self.assertEqual(pieces[1], sp.unk_id())
    self.assertEqual(pieces2[1], sp.unk_id())
    self.assertEqual(pieces, pieces2)

    pieces = self.sp_.EncodeAsPieces(unk_char, emit_unk_piece=True)
    pieces2 = self.sp_.encode(unk_char, out_type=str, emit_unk_piece=True)
    self.assertEqual(pieces[1], '<unk>')
    self.assertEqual(pieces2[1], '<unk>')
    self.assertEqual(pieces, pieces2)

    pieces = self.sp_.EncodeAsPieces(unk_char, emit_unk_piece=False)
    pieces2 = self.sp_.encode(unk_char, out_type=str, emit_unk_piece=False)
    self.assertEqual(pieces[1], unk_char)
    self.assertEqual(pieces2[1], unk_char)
    self.assertEqual(pieces, pieces2)

  def test_new_api_init(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_bos=True,
        add_eos=True,
        out_type=str,
    )
    text = 'hello world'
    pieces = ['<s>'] + self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text))

    pieces = self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text, add_bos=False, add_eos=True))

  def test_sampling(self):
    sp = self.sp_

    for out_type in [str, int, 'serialized_proto', 'immutable_proto']:
      ids = defaultdict(int)
      for n in range(100):
        out = sp.encode('hello world', out_type=out_type, enable_sampling=True)
        if type(out) is list:
          out = tuple(out)
        ++ids[out]
      self.assertGreater(len(ids), 1)

      ids2 = defaultdict(int)
      for n in range(100):
        out = sp.encode('hello world', out_type=out_type, enable_sampling=False)
        if type(out) is list:
          out = tuple(out)
        ++ids2[out]
      self.assertEqual(len(ids2), 1)

      out = sp.encode(
          ['hello world', 'this is a test'],
          out_type=out_type,
          enable_sampling=True,
      )
      self.assertEqual(len(out), 2)
      out = sp.encode(
          ['hello world', 'this is a test'],
          out_type=out_type,
          enable_sampling=False,
      )
      self.assertEqual(len(out), 2)

  def test_nbest(self):
    sp = self.sp_
    text = 'hello world'
    text2 = 'I have a pen.'

    for out_type in [str, int, 'serialized_proto', 'immutable_proto']:
      results = sp.nbest_encode(text, nbest_size=10, out_type=out_type)
      self.assertEqual(
          results, sp.NBestEncode(text, nbest_size=10, out_type=out_type)
      )

      if out_type in [str, int]:
        for n in results:
          self.assertEqual(sp.decode(n), text)

        for n in sp.decode(results):
          self.assertEqual(n, text)

      # batch test
      results = sp.nbest_encode([text, text2], nbest_size=10, out_type=out_type)
      self.assertEqual(
          results,
          sp.NBestEncode([text, text2], nbest_size=10, out_type=out_type),
      )
      self.assertEqual(len(results), 2)

      if out_type in [str, int]:
        for n in results[0]:
          self.assertEqual(sp.decode(n), text)

        for n in results[1]:
          self.assertEqual(sp.decode(n), text2)

        decoded = sp.decode(results[0])
        self.assertEqual(len(decoded), 10)
        for n in decoded:
          self.assertEqual(n, text)
        decoded = sp.decode(results[1])
        self.assertEqual(len(decoded), 10)
        for n in decoded:
          self.assertEqual(n, text2)

    self.assertEqual(
        sp.nbest_encode(text, nbest_size=10, out_type=str),
        sp.nbest_encode_as_pieces(text, nbest_size=10),
    )
    self.assertEqual(
        sp.nbest_encode(text, nbest_size=10, out_type=int),
        sp.nbest_encode_as_ids(text, nbest_size=10),
    )
    self.assertEqual(
        sp.nbest_encode(text, nbest_size=10, out_type='serialized_proto'),
        sp.nbest_encode_as_serialized_proto(text, nbest_size=10),
    )
    self.assertEqual(
        sp.nbest_encode(text, nbest_size=10, out_type='immutable_proto'),
        sp.nbest_encode_as_immutable_proto(text, nbest_size=10),
    )

  def test_sample_and_score(self):
    sp = self.sp_
    text = 'hello world'
    text2 = 'I have a pen.'
    for out_type in [str, int, 'serialized_proto', 'immutable_proto']:
      results = sp.sample_encode_and_score(
          text, wor=True, num_samples=10, out_type=out_type
      )
      results = sp.SampleEncodeAndScore(
          text, wor=False, num_samples=10, out_type=out_type
      )

      if out_type in [str, int]:
        for n in results:
          self.assertEqual(sp.decode(n[0]), text)

      results = sp.sample_encode_and_score(
          [text, text2], wor=True, num_samples=10, out_type=out_type
      )
      results = sp.SampleEncodeAndScore(
          [text, text2], wor=True, num_samples=10, out_type=out_type
      )

      if out_type in [str, int]:
        for n in results[0]:
          self.assertEqual(sp.decode(n[0]), text)
        for n in results[1]:
          self.assertEqual(sp.decode(n[0]), text2)

    sp.sample_encode_and_score_as_pieces(text, 10)
    sp.sample_encode_and_score_as_ids(text, 10)
    sp.sample_encode_and_score_as_immutable_proto(text, 10)
    sp.sample_encode_and_score_as_serialized_proto(text, 10)

  def test_valid_range(self):
    size = self.sp_.piece_size()
    funcs = [
        'IdToPiece',
        'GetScore',
        'IsUnknown',
        'IsControl',
        'IsUnused',
        'IsByte',
        'DecodeIds',
        'DecodeIdsAsSerializedProto',
    ]
    for m in funcs:
      getattr(self.sp_, m)([10, 20, 30])

    for m in funcs:
      try:
        getattr(self.sp_, m)([size])
        self.assertTrue(False)
      except:
        self.assertTrue(True)

  def test_batch(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      texts = file.readlines()

    for out_type in [str, int, 'serialized_proto', 'immutable_proto']:
      r1 = sp.encode(texts, out_type=out_type, num_threads=None)
      r2 = sp.encode(texts, out_type=out_type, num_threads=1)
      r3 = sp.encode(texts, out_type=out_type, num_threads=-1)
      r4 = sp.encode(texts, out_type=out_type, num_threads=8)
      r5 = [sp.encode(s, out_type=out_type) for s in texts]
      self.assertEqual(r1, r2)
      self.assertEqual(r1, r3)
      self.assertEqual(r1, r4)
      self.assertEqual(r1, r5)

      if out_type in [str, int]:
        d1 = sp.decode(r1, num_threads=None)
        d2 = sp.decode(r2, num_threads=1)
        d3 = sp.decode(r3, num_threads=-1)
        d4 = sp.decode(r4, num_threads=8)
        d5 = [sp.decode(s) for s in r5]

        self.assertEqual(d1, d2)
        self.assertEqual(d1, d3)
        self.assertEqual(d1, d4)
        self.assertEqual(d1, d5)

    e1 = sp.calculate_entropy(texts, alpha=1.0, num_threads=10)
    e2 = sp.CalculateEntropy(texts, alpha=1.0, num_threads=10)
    e3 = [sp.calculate_entropy(s, alpha=1.0) for s in texts]
    self.assertEqual(e1, e2)
    self.assertEqual(e1, e3)

  def test_pickle(self):
    tid = threading.get_native_id()
    with open(f'sp_{tid}.pickle', 'wb') as f:
      pickle.dump(self.sp_, f)

    id1 = self.sp_.encode('hello world.', out_type=int)

    with open(f'sp_{tid}.pickle', 'rb') as f:
      sp = pickle.load(f)

    id2 = sp.encode('hello world.', out_type=int)

    self.assertEqual(id1, id2)

  def test_global_params(self):
    spm.SetRandomGeneratorSeed(0)
    spm.SetMinLogLevel(2)
    spm.set_random_generator_seed(1)
    spm.set_min_log_level(3)

  def test_normalize(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )

    self.assertEqual('▁KADOKAWAABC', sp.normalize('ＫＡＤＯＫＡＷＡABC'))
    self.assertEqual('▁KADOKAWAABC', sp.Normalize('ＫＡＤＯＫＡＷＡABC'))

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC', with_offsets=True)
    self.assertEqual('▁KADOKAWAABC', x[0])
    self.assertEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[1])

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC'.encode('utf8'), with_offsets=True)
    self.assertEqual('▁KADOKAWAABC'.encode('utf8'), x[0])
    self.assertEqual(
        [0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[1]
    )

    self.assertEqual(
        ['▁KADOKAWAABC', '▁平成'], sp.normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )
    self.assertEqual(
        ['▁KADOKAWAABC', '▁平成'], sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )

    x = sp.Normalize(
        ['ＫＡＤＯＫＡＷＡABC'.encode('utf8'), '㍻'.encode('utf8')],
        with_offsets=True,
    )
    self.assertEqual(len(x), 2)
    self.assertEqual('▁KADOKAWAABC'.encode('utf8'), x[0][0])
    self.assertEqual(
        [0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[0][1]
    )

    x = sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'], with_offsets=True)
    self.assertEqual(len(x), 2)
    self.assertEqual('▁KADOKAWAABC', x[0][0])
    self.assertEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[0][1])

    self.assertEqual('▁平成', x[1][0])
    self.assertEqual([0, 0, 0, 1], x[1][1])

  def test_normalizer(self):
    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model')
    )

    self.assertEqual('KADOKAWAABC', sp.normalize('ＫＡＤＯＫＡＷＡABC'))
    self.assertEqual('KADOKAWAABC', sp.Normalize('ＫＡＤＯＫＡＷＡABC'))

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC'.encode('utf8'), with_offsets=True)
    self.assertEqual('KADOKAWAABC'.encode('utf8'), x[0])
    self.assertEqual([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[1])

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC', with_offsets=True)
    self.assertEqual('KADOKAWAABC', x[0])
    self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[1])

    self.assertEqual(
        ['KADOKAWAABC', '平成'], sp.normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )
    self.assertEqual(
        ['KADOKAWAABC', '平成'], sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )

    x = sp.Normalize(
        ['ＫＡＤＯＫＡＷＡABC'.encode('utf8'), '㍻'.encode('utf8')],
        with_offsets=True,
    )
    self.assertEqual(len(x), 2)
    self.assertEqual('KADOKAWAABC'.encode('utf8'), x[0][0])
    self.assertEqual([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[0][1])

    x = sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'], with_offsets=True)
    self.assertEqual(len(x), 2)
    self.assertEqual('KADOKAWAABC', x[0][0])
    self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[0][1])
    self.assertEqual('平成', x[1][0])
    self.assertEqual([0, 0, 1], x[1][1])

    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_dummy_prefix=True,
        escape_whitespaces=True,
        remove_extra_whitespaces=False,
    )
    self.assertEqual('▁hello▁▁world', sp.normalize('hello  world'))

    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_dummy_prefix=True,
        escape_whitespaces=True,
        remove_extra_whitespaces=True,
    )
    self.assertEqual('▁hello▁world', sp.normalize('  hello  world  '))

    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_dummy_prefix=False,
        escape_whitespaces=False,
        remove_extra_whitespaces=True,
    )
    self.assertEqual('hello world', sp.normalize('  hello  world  '))

  def test_normalizer_rule(self):
    sp = spm.SentencePieceNormalizer(rule_name='identity')
    self.assertEqual('ＡＢＣ', sp.Normalize('ＡＢＣ'))

    sp = spm.SentencePieceNormalizer(rule_name='nfkc_cf')
    self.assertEqual('abc', sp.Normalize('ＡＢＣ'))

  def test_override_normalize_spec(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )

    self.assertEqual(
        sp.EncodeAsPieces(' hello  world '), ['▁he', 'll', 'o', '▁world']
    )

    sp.override_normalizer_spec(add_dummy_prefix=False)
    sp.override_normalizer_spec(remove_extra_whitespaces=False)
    sp.override_normalizer_spec(escape_whitespaces=False)
    self.assertEqual(
        sp.EncodeAsPieces(' hello  world '),
        [' ', 'he', 'll', 'o', '  ', 'w', 'or', 'l', 'd', ' '],
    )


def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(TestSentencepieceProcessor))
  return suite


if __name__ == '__main__':
  unittest.main()
