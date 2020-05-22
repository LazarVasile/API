def get_best_span(span_begin_probs, span_end_probs, context_length, max_span_length):
    if len(span_begin_probs.shape) > 2 or len(span_end_probs.shape) > 2:
        raise ValueError("Input shapes must be (X,) or (1,X)")
    if len(span_begin_probs.shape) == 2:
        assert span_begin_probs.shape[0] == 1, "2D input must have an initial dimension of 1"
        span_begin_probs = span_begin_probs.flatten()
    if len(span_end_probs.shape) == 2:
        assert span_end_probs.shape[0] == 1, "2D input must have an initial dimension of 1"
        span_end_probs = span_end_probs.flatten()

    max_span_probability = 0
    best_word_span = (0, 1)

    for i, val1 in enumerate(span_begin_probs):

        for j, val2 in enumerate(span_end_probs):
            if j > context_length - 1:
                break

            if (j < i):
                continue

            if (j - i) >= max_span_length:
                break

            if val1 * val2 > max_span_probability:
                best_word_span = (i, j)
                max_span_probability = val1 * val2

    return best_word_span, max_span_probability


def get_word_char_loc_mapping(context, context_tokens):
    mapping = {}
    idx = 0
    for i, word in enumerate(context_tokens):
        id = context.find(word, idx)
        assert not id == -1, "Error occurred while mapping word index to character index.. Please report this issue on our GitHub repo."

        idx = id
        mapping[i] = id

    assert len(mapping) == len(
        context_tokens), "Error occurred while mapping word index to character index.. Please report this issue on our GitHub repo."

    return mapping

import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.engine.topology import Layer
from keras import backend as K
from keras.layers.advanced_activations import Softmax
from keras.layers import Dense, Activation, Multiply, Add, Lambda
from keras.initializers import Constant
from keras.activations import linear
from keras.layers import TimeDistributed, LSTM, Bidirectional
from keras.layers import Input
from keras.models import Model, load_model
from keras import Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adadelta
from keras.callbacks import CSVLogger, ModelCheckpoint
import os


class CombineOutputs(Layer):

    def __init__(self, **kwargs):
        super(CombineOutputs, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombineOutputs, self).build(input_shape)

    def call(self, inputs):
        span_begin_probabilities, span_end_probabilities = inputs
        return K.stack([span_begin_probabilities, span_end_probabilities],axis = 1)\

    def compute_output_shape(self, input_shape):
        number_of_tensors = len(input_shape)
        return input_shape[0][0:1] + (number_of_tensors, ) + input_shape[0][1:]

    def get_config(self):
        config = super().get_config()
        return config

class C2QAttention(Layer):

    def __init__(self, **kwargs):
        super(C2QAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(C2QAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_question = inputs
        context_to_query_attention = Softmax(axis=-1)(similarity_matrix)
        encoded_question = K.expand_dims(encoded_question, axis=1)
        return K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_question_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_question_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config

class Highway(Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        #input_shape_dense_1 = input_shape[-1]
        self.dense_1 = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(units=dim)
        self.dense_2.build(input_shape)
        self.trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dim = K.int_shape(x)[-1]
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config

class MergedContext(Layer):

    def __init__(self, **kwargs):
        super(MergedContext, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergedContext, self).build(input_shape)

    def call(self, inputs):
        encoded_context, context_to_query_attention, query_to_context_attention = inputs
        element_wise_multiply1 = encoded_context * context_to_query_attention
        element_wise_multiply2 = encoded_context * query_to_context_attention
        concatenated_tensor = K.concatenate(
            [encoded_context, context_to_query_attention, element_wise_multiply1, element_wise_multiply2], axis=-1)
        return concatenated_tensor

    def compute_output_shape(self, input_shape):
        encoded_context_shape, _, _ = input_shape
        return encoded_context_shape[:-1] + (encoded_context_shape[-1] * 4, )

    def get_config(self):
        config = super().get_config()
        return config

class Q2CAttention(Layer):

    def __init__(self, **kwargs):
        super(Q2CAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Q2CAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_context = inputs
        max_similarity = K.max(similarity_matrix, axis=-1)
        # by default, axis = -1 in Softmax
        context_to_query_attention = Softmax()(max_similarity)
        weighted_sum = K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_context, -2)
        expanded_weighted_sum = K.expand_dims(weighted_sum, 1)
        num_of_repeatations = K.shape(encoded_context)[1]
        return K.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_context_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_context_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config

class Similarity(Layer):

    def __init__(self, **kwargs):
        super(Similarity, self).__init__(**kwargs)

    def compute_similarity(self, repeated_context_vectors, repeated_query_vectors):
        element_wise_multiply = repeated_context_vectors * repeated_query_vectors
        concatenated_tensor = K.concatenate(
            [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
        dot_product = K.squeeze(K.dot(concatenated_tensor, self.kernel), axis=-1)
        return linear(dot_product + self.bias)

    def build(self, input_shape):
        word_vector_dim = input_shape[0][-1]
        weight_vector_dim = word_vector_dim * 3
        self.kernel = self.add_weight(name='similarity_weight',
                                      shape=(weight_vector_dim, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='similarity_bias',
                                    shape=(),
                                    initializer='ones',
                                    trainable=True)
        super(Similarity, self).build(input_shape)

    def call(self, inputs):
        context_vectors, query_vectors = inputs
        num_context_words = K.shape(context_vectors)[1]
        num_query_words = K.shape(query_vectors)[1]
        context_dim_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
        query_dim_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
        repeated_context_vectors = K.tile(K.expand_dims(context_vectors, axis=2), context_dim_repeat)
        repeated_query_vectors = K.tile(K.expand_dims(query_vectors, axis=1), query_dim_repeat)
        similarity_matrix = self.compute_similarity(repeated_context_vectors, repeated_query_vectors)
        return similarity_matrix

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        num_context_words = input_shape[0][1]
        num_query_words = input_shape[1][1]
        return (batch_size, num_context_words, num_query_words)

    def get_config(self):
        config = super().get_config()
        return config

class SpanBegin(Layer):

    def __init__(self, **kwargs):
        super(SpanBegin, self).__init__(**kwargs)

    def build(self, input_shape):
        last_dim = input_shape[0][-1] + input_shape[1][-1]
        input_shape_dense_1 = input_shape[0][:-1] + (last_dim, )
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.dense_1.trainable_weights
        super(SpanBegin, self).build(input_shape)

    def call(self, inputs):
        merged_context, modeled_passage = inputs
        span_begin_input = K.concatenate([merged_context, modeled_passage])
        span_begin_weights = TimeDistributed(self.dense_1)(span_begin_input)
        span_begin_probabilities = Softmax()(K.squeeze(span_begin_weights, axis=-1))
        return span_begin_probabilities

    def compute_output_shape(self, input_shape):
        merged_context_shape, _ = input_shape
        return merged_context_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config

class SpanEnd(Layer):

    def __init__(self, **kwargs):
        super(SpanEnd, self).__init__(**kwargs)

    def build(self, input_shape):
        emdim = input_shape[0][-1] // 2
        input_shape_bilstm_1 = input_shape[0][:-1] + (emdim*14, )
        self.bilstm_1 = Bidirectional(LSTM(emdim, return_sequences=True))
        self.bilstm_1.build(input_shape_bilstm_1)
        input_shape_dense_1 = input_shape[0][:-1] + (emdim*10, )
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.bilstm_1.trainable_weights + self.dense_1.trainable_weights
        super(SpanEnd, self).build(input_shape)

    def call(self, inputs):
        encoded_passage, merged_context, modeled_passage, span_begin_probabilities = inputs
        weighted_sum = K.sum(K.expand_dims(span_begin_probabilities, axis=-1) * modeled_passage, -2)
        passage_weighted_by_predicted_span = K.expand_dims(weighted_sum, axis=1)
        tile_shape = K.concatenate([[1], [K.shape(encoded_passage)[1]], [1]], axis=0)
        passage_weighted_by_predicted_span = K.tile(passage_weighted_by_predicted_span, tile_shape)
        multiply1 = modeled_passage * passage_weighted_by_predicted_span
        span_end_representation = K.concatenate(
            [merged_context, modeled_passage, passage_weighted_by_predicted_span, multiply1])

        span_end_representation = self.bilstm_1(span_end_representation)

        span_end_input = K.concatenate([merged_context, span_end_representation])

        span_end_weights = TimeDistributed(self.dense_1)(span_end_input)

        span_end_probabilities = Softmax()(K.squeeze(span_end_weights, axis=-1))
        return span_end_probabilities

    def compute_output_shape(self, input_shape):
        _, merged_context_shape, _, _ = input_shape
        return merged_context_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus=None):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

from scripturi.processing import accuracy, negative_avg_log_error, tokenize, MagnitudeVectors, load_data_generators

class BidirectionalAttentionFlow():

    def __init__(self, emdim, max_passage_length=None, max_query_length=None, num_highway_layers=2, num_decoders=1, encoder_dropout=0, decoder_dropout=0):
        self.emdim = emdim
        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        passage_input = Input(shape=(self.max_passage_length, emdim), dtype='float32', name="passage_input")
        question_input = Input(shape=(self.max_query_length, emdim), dtype='float32', name="question_input")

        question_embedding = question_input
        passage_embedding = passage_input
        for i in range(num_highway_layers):
            highway_layer = Highway(name='highway_{}'.format(i))
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_embedding)
            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_embedding)

        encoder_layer = Bidirectional(LSTM(emdim, recurrent_dropout=encoder_dropout,
                                           return_sequences=True), name='bidirectional_encoder')
        encoded_question = encoder_layer(question_embedding)
        encoded_passage = encoder_layer(passage_embedding)

        similarity_matrix = Similarity(name='similarity_layer')([encoded_passage, encoded_question])

        context_to_query_attention = C2QAttention(name='context_to_query_attention')([
            similarity_matrix, encoded_question])
        query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
            similarity_matrix, encoded_passage])

        merged_context = MergedContext(name='merged_context')(
            [encoded_passage, context_to_query_attention, query_to_context_attention])

        modeled_passage = merged_context
        for i in range(num_decoders):
            hidden_layer = Bidirectional(LSTM(emdim, recurrent_dropout=decoder_dropout,
                                              return_sequences=True), name='bidirectional_decoder_{}'.format(i))
            modeled_passage = hidden_layer(modeled_passage)

        span_begin_probabilities = SpanBegin(name='span_begin')([merged_context, modeled_passage])
        span_end_probabilities = SpanEnd(name='span_end')(
            [encoded_passage, merged_context, modeled_passage, span_begin_probabilities])

        output = CombineOutputs(name='combine_outputs')([span_begin_probabilities, span_end_probabilities])

        model = Model([passage_input, question_input], [output])

        model.summary()

        try:
            model = ModelMGPU(model)
        except:
            pass

        adadelta = Adadelta(lr=0.01)
        model.compile(loss=negative_avg_log_error, optimizer=adadelta, metrics=[accuracy])

        self.model = model

    def load_bidaf(self, path):
        custom_objects = {
            'Highway': Highway,
            'Similarity': Similarity,
            'C2QAttention': C2QAttention,
            'Q2CAttention': Q2CAttention,
            'MergedContext': MergedContext,
            'SpanBegin': SpanBegin,
            'SpanEnd': SpanEnd,
            'CombineOutputs': CombineOutputs,
            'negative_avg_log_error': negative_avg_log_error,
            'accuracy': accuracy
        }

        self.model = load_model(path, custom_objects=custom_objects)

    def train_model(self, train_generator, steps_per_epoch=None, epochs=1, validation_generator=None,
                    validation_steps=None, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0,
                    save_history=False, save_model_per_epoch=False):

        saved_items_dir = os.path.join(os.path.dirname(__file__), 'saved_items')
        if not os.path.exists(saved_items_dir):
            os.makedirs(saved_items_dir)

        callbacks = []

        if save_history:
            history_file = os.path.join(saved_items_dir, 'history')
            csv_logger = CSVLogger(history_file, append=True)
            callbacks.append(csv_logger)

        if save_model_per_epoch:
            save_model_file = os.path.join(saved_items_dir, 'bidaf_2.h5')
            checkpointer = ModelCheckpoint(filepath=save_model_file, verbose=1)
            callbacks.append(checkpointer)

        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           callbacks=callbacks, validation_data=validation_generator,
                                           validation_steps=validation_steps, workers=workers,
                                           use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                           initial_epoch=initial_epoch)
        if not save_model_per_epoch:
            self.model.save(os.path.join(saved_items_dir, 'bidaf.h5'))

        return history, self.model

    def predict_ans(self, passage, question, squad_version=1.1, max_span_length=25, do_lowercase=True,
                    return_char_loc=False, return_confidence_score=False):

        if type(passage) == list:
            assert all(type(pas) == str for pas in passage), "Input 'passage' must be of type 'string'"

            passage = [pas.strip() for pas in passage]
            contexts = []
            for pas in passage:
                context_tokens = tokenize(pas, do_lowercase)
                contexts.append(context_tokens)

            if do_lowercase:
                original_passage = [pas.lower() for pas in passage]
            else:
                original_passage = passage

        elif type(passage) == str:
            passage = passage.strip()
            context_tokens = tokenize(passage, do_lowercase)
            contexts = [context_tokens, ]

            if do_lowercase:
                original_passage = [passage.lower(), ]
            else:
                original_passage = [passage, ]

        else:
            raise TypeError("Input 'passage' must be either a 'string' or 'list of strings'")

        assert type(passage) == type(
            question), "Both 'passage' and 'question' must be either 'string' or a 'list of strings'"

        if type(question) == list:
            assert all(type(ques) == str for ques in question), "Input 'question' must be of type 'string'"
            assert len(passage) == len(
                question), "Both lists (passage and question) must contain same number of elements"

            questions = []
            for ques in question:
                question_tokens = tokenize(ques, do_lowercase)
                questions.append(question_tokens)

        elif type(question) == str:
            question_tokens = tokenize(question, do_lowercase)
            questions = [question_tokens, ]

        else:
            raise TypeError("Input 'question' must be either a 'string' or 'list of strings'")

        vectors = MagnitudeVectors(self.emdim).load_vectors()
        context_batch = vectors.query(contexts, self.max_passage_length)
        question_batch = vectors.query(questions, self.max_query_length)

        y = self.model.predict([context_batch, question_batch])
        y_pred_start = y[:, 0, :]
        y_pred_end = y[:, 1, :]

        # clearing the session releases memory by removing the model from memory
        # using this, you will need to load model every time before prediction
        # K.clear_session()

        batch_answer_span = []
        batch_confidence_score = []
        for sample_id in range(len(contexts)):
            answer_span, confidence_score = get_best_span(y_pred_start[sample_id, :], y_pred_end[sample_id, :], len(contexts[sample_id]), max_span_length)
            batch_answer_span.append(answer_span)
            batch_confidence_score.append(confidence_score)

        answers = []
        for index, answer_span in enumerate(batch_answer_span):
            context_tokens = contexts[index]
            start, end = answer_span[0], answer_span[1]

            # word index to character index mapping
            mapping = get_word_char_loc_mapping(original_passage[index], context_tokens)

            char_loc_start = mapping[start]
            # [1] => char_loc_end is set to point to one more character after the answer
            char_loc_end = mapping[end] + len(context_tokens[end])
            # [1] will help us getting a perfect slice without unnecessary increments/decrements
            ans = original_passage[index][char_loc_start:char_loc_end]

            return_dict = {
                "answer": ans,
            }

            if return_char_loc:
                return_dict["char_loc_start"] = char_loc_start
                return_dict["char_loc_end"] = char_loc_end - 1

            if return_confidence_score:
                return_dict["confidence_score"] = batch_confidence_score[index]

            answers.append(return_dict)

        if type(passage) == list:
            return answers
        else:
            return answers[0]


""" emdim = 300
max_passage_length = None
max_query_length = None
num_highway_layers = 1
num_decoders = 1
encoder_dropout = 0.0
decoder_dropout = 0.0

batch_size = 2
shuffle_samples = False
steps_per_epochs = None
epochs = 50
validation_steps = None
workers = 1
use_multiprocessing = False
shuffle_batch = False
save_history = False
save_model_per_epoch = True

#from processing import data_download_and_preprocess

#data_download_and_preprocess(do_lowercase=True)

bidaf_model = BidirectionalAttentionFlow(emdim=emdim, max_passage_length=max_passage_length,
                                             max_query_length=max_query_length,
                                             num_highway_layers=num_highway_layers, num_decoders=num_decoders,
                                             encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout)

print("Model created!") """

""" train_generator, validation_generator = load_data_generators(batch_size=batch_size, emdim=emdim,
                                                                     max_passage_length=max_passage_length,
                                                                     max_query_length=max_query_length,
                                                                     shuffle=shuffle_samples)

print("Dataset loaded!")

#bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), 'saved_items', model_name))

#print("Model loaded!")

#bidaf_model.model.compile(loss=negative_avg_log_error, optimizer='adadelta', metrics=[accuracy])

#print("Model compiled!")

bidaf_model.train_model(train_generator, steps_per_epoch=steps_per_epochs, epochs=epochs,
                                validation_generator=validation_generator, validation_steps=validation_steps,
                                workers=workers, use_multiprocessing=use_multiprocessing,
                                shuffle=shuffle_batch, save_history=save_history,
                                save_model_per_epoch=save_model_per_epoch)


print("Training Completed!") """


""" model_name = "bidaf.h5"

bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', model_name))

print("Model loaded!")

passage = "Tesla, Inc. este un constructor de automobile electrice de înaltă performanță, din Silicon Valley. Tesla a primit o atenție deosebită când au lansat modelul de producție Tesla Roadster, prima mașină sport 100 electrică. A doua mașina produsă de Tesla este Model S, 100 electric sedan de lux."

question = "Care este a doua mașina produsă de Tesla?"

answer = bidaf_model.predict_ans(passage, question)

print("Question: ", question)
print("Predicted answer:", answer) """


