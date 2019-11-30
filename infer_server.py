"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# import os
import random
import time
# from time import gmtime, strftime
# import wave
import argparse
import functools
import struct
import SocketServer

import numpy as np

import paddle.fluid as fluid
from data_utils.data import DataGenerator
from data_utils.utility import read_manifest
from model_utils.model import DeepSpeech2Model
from model_utils.model_check import check_cuda, check_version
# from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments


class AsrTCPServer(SocketServer.TCPServer):
    """The ASR TCP Server."""

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 speech_save_dir,
                 audio_process_handler,
                 bind_and_activate=True):
        self.speech_save_dir = speech_save_dir
        self.audio_process_handler = audio_process_handler
        SocketServer.TCPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate=True)

class AsrRequestHandler(SocketServer.BaseRequestHandler):
    """The ASR request handler."""

    def handle(self):
        # receive data through TCP socket
        chunk = self.request.recv(1024)
        target_len = struct.unpack('>i', chunk[:4])[0]
        data = chunk[4:]
        while len(data) < target_len:
            chunk = self.request.recv(1024)
            data += chunk
        # write to file
        # filename = self._write_to_file(data)
        print('data:', type(data), data)
        filename = data

        print("Received utterance[length=%d] from %s, saved to %s." % (len(data), self.client_address[0], filename))
        start_time = time.time()
        transcript = self.server.audio_process_handler(filename)
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" % (finish_time - start_time, transcript))
        self.request.sendall(transcript.encode('utf-8'))

    # def _write_to_file(self, data):
        # # prepare save dir and filename
        # if not os.path.exists(self.server.speech_save_dir):
            # os.mkdir(self.server.speech_save_dir)
        # timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        # out_filename = os.path.join(
            # self.server.speech_save_dir,
            # timestamp + "_" + self.client_address[0] + ".wav")
        # # write to wav file
        # file = wave.open(out_filename, 'wb')
        # file.setnchannels(1)
        # file.setsampwidth(4)
        # file.setframerate(16000)
        # file.writeframes(data)
        # file.close()
        # return out_filename


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('host_port',        int,    8086,    "Server's IP port.")
add_arg('host_ip',          str, 'localhost', "Server's IP address.")
add_arg('num_samples',      int,    10,     "# of samples to infer.")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.5,    "Coef of LM for beam search.")
add_arg('beta',             float,  0.3,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('speech_save_dir',  str,
        'demo_cache',
        "Directory to save demo audios.")
add_arg('infer_manifest',   str,
        # 'data/librispeech/manifest.dev-clean',
        'data/librispeech/manifest.test-clean',
        "Filepath of manifest to infer.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('lang_model_path',  str,
        'models/lm/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('model_path',       str,
        './checkpoints/libri/step_final',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()


def warm_up_test(audio_process_handler,
                 manifest_path,
                 num_test_cases,
                 random_seed=0):
    """Warming-up test."""
    manifest = read_manifest(manifest_path)
    rng = random.Random(random_seed)
    samples = rng.sample(manifest, num_test_cases)
    for idx, sample in enumerate(samples):
        print("Warm-up Test Case %d: %s" % (idx, sample['audio_filepath']))
        start_time = time.time()
        transcript = audio_process_handler(sample['audio_filepath'])
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" % (finish_time - start_time, transcript))

def infer():
    """Inference for DeepSpeech2."""

    # check if set use_gpu=True in paddlepaddle cpu version
    check_cuda(args.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    print('a1')
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        keep_transcription_text=True,
        place = place,
        is_training = False)
    
    print('a2')
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.infer_manifest,
        batch_size=args.num_samples,
        sortagrad=False,
        shuffle_method=None)
    infer_data = next(batch_reader())

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        share_rnn_weights=args.share_rnn_weights,
        place=place,
        init_from_pretrained_model=args.model_path)

    # decoders only accept string encoded in utf-8
    print('a3')
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    if args.decoding_method == "ctc_beam_search":
        ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path,
                                  vocab_list)

    # if args.decoding_method == "ctc_greedy":
        # ds2_model.logger.info("start inference ...")
        # probs_split = ds2_model.infer_batch_probs(
            # infer_data=infer_data,
            # feeding_dict=data_generator.feeding)

        # result_transcripts = ds2_model.decode_batch_greedy(
            # probs_split=probs_split,
            # vocab_list=vocab_list)
    # else:
        # ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path,
                                  # vocab_list)
        # ds2_model.logger.info("start inference ...")
        # probs_split= ds2_model.infer_batch_probs(
            # infer_data=infer_data,
            # feeding_dict=data_generator.feeding)

        # result_transcripts= ds2_model.decode_batch_beam_search(
            # probs_split=probs_split,
            # beam_alpha=args.alpha,
            # beam_beta=args.beta,
            # beam_size=args.beam_size,
            # cutoff_prob=args.cutoff_prob,
            # cutoff_top_n=args.cutoff_top_n,
            # vocab_list=vocab_list,
            # num_processes=args.num_proc_bsearch)

    def file_to_transcript(filename):
        feature = data_generator.process_utterance(filename, "")
        audio_len = feature[0].shape[1]
        mask_shape0 = (feature[0].shape[0] - 1) // 2 + 1
        mask_shape1 = (feature[0].shape[1] - 1) // 3 + 1
        mask_max_len = (audio_len - 1) // 3 + 1
        mask_ones = np.ones((mask_shape0, mask_shape1))
        mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
        mask = np.repeat(
            np.reshape(
                np.concatenate((mask_ones, mask_zeros), axis=1),
                (1, mask_shape0, mask_max_len)),
            32,
            axis=0)
        feature = (np.array([feature[0]]).astype('float32'),
                   None,
                   np.array([audio_len]).astype('int64').reshape([-1,1]),
                   np.array([mask]).astype('float32'))
        probs_split = ds2_model.infer_batch_probs(
            infer_data=feature,
            feeding_dict=data_generator.feeding)

        if args.decoding_method == "ctc_greedy":
            result_transcript = ds2_model.decode_batch_greedy(
                probs_split=probs_split,
                vocab_list=vocab_list)
        else:
            result_transcript = ds2_model.decode_batch_beam_search(
                probs_split=probs_split,
                beam_alpha=args.alpha,
                beam_beta=args.beta,
                beam_size=args.beam_size,
                cutoff_prob=args.cutoff_prob,
                cutoff_top_n=args.cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=1)
        print('result_transcript:', result_transcript)
        return result_transcript[0]

    # error_rate_func = cer if args.error_rate_type == 'cer' else wer
    # target_transcripts = infer_data[1]
    # for target, result in zip(target_transcripts, result_transcripts):
        # print("\nTarget Transcription: %s\nOutput Transcription: %s" %
              # (target, result))
        # print("Current error rate [%s] = %f" %
              # (args.error_rate_type, error_rate_func(target, result)))

    # ds2_model.logger.info("finish inference")


    # warming up with utterrances sampled from Librispeech
    print('-----------------------------------------------------------')
    print('Warming up ...')
    warm_up_test(
        audio_process_handler=file_to_transcript,
        manifest_path=args.infer_manifest,
        num_test_cases=3)
    # file_to_transcript('/home/chris/git/speech_to_text_test/data/audio/go-forward-two-meters-and-then-stop.rate16k-mono.wav')
    print('-----------------------------------------------------------')

    # start the server
    server = AsrTCPServer(
        server_address=(args.host_ip, args.host_port),
        RequestHandlerClass=AsrRequestHandler,
        speech_save_dir=args.speech_save_dir,
        audio_process_handler=file_to_transcript)
    print("ASR Server Started.")
    server.serve_forever()

def main():
    print_arguments(args)
    infer()


if __name__ == '__main__':
    main()
