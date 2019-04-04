import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import datetime
import pickle

from utils import shared, set_values, get_name
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward
from optimization import Optimization


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            # Model location
            model_id = str(datetime.datetime.now())+"fold"+str(parameters['fold'])
            model_path = os.path.join(models_path, model_id)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                pickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = pickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            pickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              bins,
              with_eeg_gaze,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Number of total reading time features
        if with_eeg_gaze:
            n_bins = bins

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')
        if with_eeg_gaze:
            # create embedding vectors for all cognitive features
            tfd_ids = T.ivector(name='tfd_ids')
            n_fix_ids = T.ivector(name='n_fix_ids')
            ffd_ids = T.ivector(name='ffd_ids')
            fpd_ids = T.ivector(name='fpd_ids')
            fix_prob_ids = T.ivector(name='fix_prob_ids')
            n_ref_ids = T.ivector(name='n_ref_ids')
            rrp_ids = T.ivector(name='rrp_ids')
            mfd_ids = T.ivector(name='mfd_ids')
            rfd_ids = T.ivector(name='rfd_ids')
            wm2_fix_prob_ids = T.ivector(name='wm2_fix_prob_ids')
            wm1_fix_prob_ids = T.ivector(name='wm1_fix_prob_ids')
            wp1_fix_prob_ids = T.ivector(name='wp1_fix_prob_ids')
            wp2_fix_prob_ids = T.ivector(name='wp2_fix_prob_ids')
            wm2_fix_dur_ids = T.ivector(name='wm2_fix_dur_ids')
            wm1_fix_dur_ids = T.ivector(name='wm1_fix_dur_ids')
            wp1_fix_dur_ids = T.ivector(name='wp1_fix_dur_ids')
            wp2_fix_dur_ids = T.ivector(name='wp2_fix_dur_ids')
            ffd_t1_ids = T.ivector(name='ffd_t1_ids')
            ffd_t2_ids = T.ivector(name='ffd_t2_ids')
            ffd_a1_ids = T.ivector(name='ffd_a1_ids')
            ffd_a2_ids = T.ivector(name='ffd_a2_ids')
            ffd_b1_ids = T.ivector(name='ffd_b1_ids')
            ffd_b2_ids = T.ivector(name='ffd_b2_ids')
            ffd_g1_ids = T.ivector(name='ffd_g1_ids')
            ffd_g2_ids = T.ivector(name='ffd_g2_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print('Loading pretrained embeddings from %s...' % pre_emb)
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print('WARNING: %i invalid lines' % emb_invalid)
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in range(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                word_layer.embeddings.set_value(new_weights)
                print('Loaded %i pretrained embeddings.' % len(pretrained))
                print(('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      ))
                print(('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      ))

        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))

        #
        # Eye tracking features
        #
        if with_eeg_gaze:

            with_eeg_gaze = 1

            input_dim += with_eeg_gaze
            tfd_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='tfd_layer')
            inputs.append(tfd_layer.link(tfd_ids))

            input_dim += with_eeg_gaze
            n_fix_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='n_fix_layer')
            inputs.append(n_fix_layer.link(n_fix_ids))

            input_dim += with_eeg_gaze
            ffd_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_layer')
            inputs.append(ffd_layer.link(ffd_ids))

            input_dim += with_eeg_gaze
            fpd_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='fpd_layer')
            inputs.append(fpd_layer.link(fpd_ids))

            input_dim += with_eeg_gaze
            fix_prob_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='fix_prob_layer')
            inputs.append(fix_prob_layer.link(fix_prob_ids))

            input_dim += with_eeg_gaze
            n_ref_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='n_ref_layer')
            inputs.append(n_ref_layer.link(n_ref_ids))

            input_dim += with_eeg_gaze
            rrp_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='rrp_layer')
            inputs.append(rrp_layer.link(rrp_ids))

            input_dim += with_eeg_gaze
            mfd_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='mfd_layer')
            inputs.append(mfd_layer.link(mfd_ids))

            input_dim += with_eeg_gaze
            rfd_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='rfd_layer')
            inputs.append(rfd_layer.link(rfd_ids))

            input_dim += with_eeg_gaze
            wm2_fix_prob_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wm2_fix_prob_layer')
            inputs.append(wm2_fix_prob_layer.link(wm2_fix_prob_ids))

            input_dim += with_eeg_gaze
            wm1_fix_prob_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wm1_fix_prob_layer')
            inputs.append(wm1_fix_prob_layer.link(wm1_fix_prob_ids))

            input_dim += with_eeg_gaze
            wp1_fix_prob_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wp1_fix_prob_layer')
            inputs.append(wp1_fix_prob_layer.link(wp1_fix_prob_ids))

            input_dim += with_eeg_gaze
            wp2_fix_prob_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wp2_fix_prob_layer')
            inputs.append(wp2_fix_prob_layer.link(wp2_fix_prob_ids))

            input_dim += with_eeg_gaze
            wm2_fix_dur_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wm2_fix_dur_layer')
            inputs.append(wm2_fix_dur_layer.link(wm2_fix_dur_ids))

            input_dim += with_eeg_gaze
            wm1_fix_dur_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wm1_fix_dur_layer')
            inputs.append(wm1_fix_dur_layer.link(wm1_fix_dur_ids))

            input_dim += with_eeg_gaze
            wp1_fix_dur_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wp1_fix_dur_layer')
            inputs.append(wp1_fix_dur_layer.link(wp1_fix_dur_ids))

            input_dim += with_eeg_gaze
            wp2_fix_dur_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='wp2_fix_dur_layer')
            inputs.append(wp2_fix_dur_layer.link(wp2_fix_dur_ids))

            input_dim += with_eeg_gaze
            ffd_t1_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_t1_layer')
            inputs.append(ffd_t1_layer.link(ffd_t1_ids))

            input_dim += with_eeg_gaze
            ffd_t2_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_t2_layer')
            inputs.append(ffd_t2_layer.link(ffd_t2_ids))

            input_dim += with_eeg_gaze
            ffd_a1_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_a1_layer')
            inputs.append(ffd_a1_layer.link(ffd_a1_ids))

            input_dim += with_eeg_gaze
            ffd_a2_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_a2_layer')
            inputs.append(ffd_a2_layer.link(ffd_a2_ids))

            input_dim += with_eeg_gaze
            ffd_b1_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_b1_layer')
            inputs.append(ffd_b1_layer.link(ffd_b1_ids))

            input_dim += with_eeg_gaze
            ffd_b2_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_b2_layer')
            inputs.append(ffd_b2_layer.link(ffd_b2_ids))

            input_dim += with_eeg_gaze
            ffd_g1_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_g1_layer')
            inputs.append(ffd_g1_layer.link(ffd_g1_ids))

            input_dim += with_eeg_gaze
            ffd_g2_layer = EmbeddingLayer(n_bins, with_eeg_gaze, name='ffd_g2_layer')
            inputs.append(ffd_g2_layer.link(ffd_g2_ids))


        # Prepare final input
        inputs = T.concatenate(inputs, axis=1) if len(inputs) != 1 else inputs[0]

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')

            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            np.random.seed(1901)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            np.random.seed(1901)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        if with_eeg_gaze:
            self.add_component(tfd_layer)
            params.extend(tfd_layer.params)
            self.add_component(n_fix_layer)
            params.extend(n_fix_layer.params)
            self.add_component(ffd_layer)
            params.extend(ffd_layer.params)
            self.add_component(fpd_layer)
            params.extend(fpd_layer.params)
            self.add_component(fix_prob_layer)
            params.extend(fix_prob_layer.params)
            self.add_component(n_ref_layer)
            params.extend(n_ref_layer.params)
            self.add_component(rrp_layer)
            params.extend(rrp_layer.params)
            self.add_component(mfd_layer)
            params.extend(mfd_layer.params)
            self.add_component(rfd_layer)
            params.extend(rfd_layer.params)
            self.add_component(wm2_fix_prob_layer)
            params.extend(wm2_fix_prob_layer.params)
            self.add_component(wm1_fix_prob_layer)
            params.extend(wm1_fix_prob_layer.params)
            self.add_component(wp1_fix_prob_layer)
            params.extend(wp1_fix_prob_layer.params)
            self.add_component(wp2_fix_prob_layer)
            params.extend(wp2_fix_prob_layer.params)
            self.add_component(wm2_fix_dur_layer)
            params.extend(wm2_fix_dur_layer.params)
            self.add_component(wm1_fix_dur_layer)
            params.extend(wm1_fix_dur_layer.params)
            self.add_component(wp1_fix_dur_layer)
            params.extend(wp1_fix_dur_layer.params)
            self.add_component(wp2_fix_dur_layer)
            params.extend(wp2_fix_dur_layer.params)
            self.add_component(ffd_t1_layer)
            params.extend(ffd_t1_layer.params)
            self.add_component(ffd_t2_layer)
            params.extend(ffd_t2_layer.params)
            self.add_component(ffd_a1_layer)
            params.extend(ffd_a1_layer.params)
            self.add_component(ffd_a2_layer)
            params.extend(ffd_a2_layer.params)
            self.add_component(ffd_b1_layer)
            params.extend(ffd_b1_layer.params)
            self.add_component(ffd_b2_layer)
            params.extend(ffd_b2_layer.params)
            self.add_component(ffd_g1_layer)
            params.extend(ffd_g1_layer.params)
            self.add_component(ffd_g2_layer)
            params.extend(ffd_g2_layer.params)

        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        if with_eeg_gaze:
            eval_inputs.append(tfd_ids)
            eval_inputs.append(n_fix_ids)
            eval_inputs.append(ffd_ids)
            eval_inputs.append(fpd_ids)
            eval_inputs.append(fix_prob_ids)
            eval_inputs.append(n_ref_ids)
            eval_inputs.append(rrp_ids)
            eval_inputs.append(mfd_ids)
            eval_inputs.append(rfd_ids)
            eval_inputs.append(wm2_fix_prob_ids)
            eval_inputs.append(wm1_fix_prob_ids)
            eval_inputs.append(wp1_fix_prob_ids)
            eval_inputs.append(wp2_fix_prob_ids)
            eval_inputs.append(wm2_fix_dur_ids)
            eval_inputs.append(wm1_fix_dur_ids)
            eval_inputs.append(wp1_fix_dur_ids)
            eval_inputs.append(wp2_fix_dur_ids)
            eval_inputs.append(ffd_t1_ids)
            eval_inputs.append(ffd_t2_ids)
            eval_inputs.append(ffd_a1_ids)
            eval_inputs.append(ffd_a2_ids)
            eval_inputs.append(ffd_b1_ids)
            eval_inputs.append(ffd_b2_ids)
            eval_inputs.append(ffd_g1_ids)
            eval_inputs.append(ffd_g2_ids)

        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print('Compiling...')
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        return f_train, f_eval
