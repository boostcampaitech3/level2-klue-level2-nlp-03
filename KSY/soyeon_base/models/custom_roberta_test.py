#################################
### custom model 테스트를 위한 file
#################################

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import RobertaModel
from transformers import (
                    AutoTokenizer,
                    AutoConfig,
                    AutoModelForSequenceClassification,
                    RobertaPreTrainedModel,
                    PreTrainedModel
                    )

from transformers.models.roberta.modeling_roberta import (RobertaLayer,
                    RobertaSelfAttention,RobertaSelfOutput,RobertaAttention,
                    RobertaEncoder,RobertaEncoder,RobertaIntermediate)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

#####################################
# TODO 검색하셔서, 해당 부분 중심으로 확인하시면 될 것 같습니다.

# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # TODO: RobertaModel에서 Pooler를 사용할 경우, hidden states의 첫번째 토큰 만 사용해서 Linear, activation을 거칩니다.
        # hidden_states의 shape은 아마 [batch_size, sequence_len, feature_dim]이고
        # hidden_states[0] 은 [batch_size, feature_dim]이 될 것같은데 확인 필요합니다.
        # 향후, entity embedding도 classifier에 넣으려면 해당 index를 통해 그 벡터값만 꺼내서 fc layer에 넣고 concat 하는 등이 필요할 것 같습니다.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    Masking 가능하도록 옵션 추가
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        # 1. dimension check :
        # embedding dimension 은 vocab추가 시 config.vocab_size 변동
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # checkpoint에서 불러오는 parameter 때문에 type_vocab_size 를 2로 늘려주면 에러 발생!
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size) # roberta-large 1 / bert-base 2
                                                    # 0,1
        # 따로 선언해서 넣어주는 방법 사용 -> 대신 initialization 중요할듯?
        self.use_entity_embedding = config.use_entity_embedding
        if config.use_entity_embedding:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('@@@ you are using entity embedding @@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            self.entity_type_embeddings = nn.Embedding(config.entity_emb_size, config.hidden_size)
        else:
            print('---------------------------------------')
            print('@@@  N O T  using entity embedding @@@@')
            print('---------------------------------------')

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        # 추후 entity embedding 값을 선언하고, forward 부분에서 더해주는 것 필요할듯

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, e1_mask=None, e2_mask=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.use_entity_embedding:
            entity_type_embeddings = self.entity_type_embeddings(e1_mask+e2_mask)
            embeddings += entity_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

# Encoder는 크게 변동할 사항이 없을 것 같습니다.
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# 참고하는 코드들은 RobertaModel 은 크게 손대지 않았던데
# 참고 코드 https://github.com/boostcampaitech2/klue-level2-nlp-14/blob/5154eca96ee9b7f17e5544c54d578ae1f10df401/solution/models/modeling_roberta.py#L10
# 아마 손댈 필요가 있지 않을까 싶습니다.

## forward 인자, embedding, classifier에 들어가는 부분 수정
class customRobertaModel(RobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    # TODO : add_pooling_layer boolean 값에 따라 아래의 pooler가 생성/미생성됩니다.
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        # TODO : RobertaPooler 선택 부분
        # add_pooling_layer의 default는 True지만 RobertaPretrainedModel class에서 설정합니다.
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # 참고 : 원래 github에서 긁어온 RobertaPreTrainedModel 은 post_init() 메소드가 PreTrainedModel에 있기 때문에 돼야하는데
        # 없는 method라고 떠서 추가로 가져왔습니다.
        self.post_init()

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        # self._backward_compatibility_gradient_checkpointing()

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        _init_weights= True
        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        e1_mask : Optional[torch.Tensor] =None, # 여기
        e2_mask : Optional[torch.Tensor] =None, # 여기
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            e1_mask = e1_mask,
            e2_mask = e2_mask
        )


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # TODO: encoder_outputs 은 BaseModelOutputWithPastAndCrossAttentions 클래스로 반환됩니다.
        # 그래서 어떤 값들이 리턴되고 어떤걸 선택해서 sequence output으로 사용되는지 확인할 필요가 있습니다.

        sequence_output = encoder_outputs[0]
        # breakpoint()
        # TODO: self.pooler가 있고 없고에 따른 pooled_output dimension
        # breakpoint()
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        # 마찬가지로 특정 class 형태로 아웃풋이 반환됩니다.
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# 수정 중
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # config.hidden_size -> 2*config.hidden_size로 해봄
        # self.dense = nn.Linear(3*config.hidden_size, config.hidden_size )
        #
        self.entity_fc = FCLayer(config.hidden_size, config.hidden_size)
        self.cls_fc = FCLayer(config.hidden_size, config.hidden_size)
        # 0.1
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.out_proj = FCLayer(3*config.hidden_size, config.num_labels,
                                classifier_dropout, use_activation=False)

    def entity_average(self, hidden_output, e_mask):
        # 코드 참고
        # https://github.com/monologg/R-BERT/blob/master/model.py
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, features, e1_mask, e2_mask, avg = False, **kwargs):
        # avg or use cls token


        e1_h = self.entity_average(features, e1_mask)
        e2_h = self.entity_average(features, e2_mask)
        pooled_output =  features[:, 0, :]

        pooled_output = self.cls_fc(pooled_output)
        e1_h = self.entity_fc(e1_h)
        e2_h = self.entity_fc(e2_h)

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        x = self.out_proj(concat_h)

        """ 수정님과 회의
        # subj, obj = features[:,e1_mask] , feature[:,e2_mask]
        # subj [bs, subj_길이, hidden_dim]
        # subj.mean() -> [bs, hidden_dim]

        # obj [bs, subj_길이, hidden_dim]
        # obj.mean() -> [bs, hidden_dim]
        """

        return x

class RobertaClassificationLSTMHead(nn.Module):
    """LSTM(bidirectional model"""
    # TODO: LSTM의 hidden dimension configuration @ 소연
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size,
                            batch_first=True, bidirectional=True )

        self.fc1 = nn.Linear(2*config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, all = True, **kwargs):
        if all:
            # LSTM에 전체 문장 token embedding 을 넣을 경우임
            # 일단 현재는 LSTM 사용할 경우 모든 sequence 받는걸로 설정.
            x = features
        else:
            # LSTM으로 들어갈 shape 때문에
            # [CLS] embedding 만 LSTM에 들어간다면 unsqueeze 처리 필요
            # 단, 이것만 LSTM 처리하는건 FC layer 넣는거랑 별반 다를게 없음.
            # 따라서 entity embedding을 더 입력으로 받거나 아니면 위에 처럼
            # 전체 token embedding sequence  넣는 것이 필요
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
            x = x.unsqueeze(dim=1) # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        x = self.dropout(x)

        # 향후 TODO: lstm의 output, hidden dim 나오는 방식
        output, hidden  = self.lstm(x)
        x = output[:,-1,:] # last output
        x = torch.tanh(x) # activation
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RobertaClassificationBidirectionalLSTMHead(nn.Module):
    """Modified LSTM by seyeonpark
    LSTM의 Bidrectional output 고려한 mixing
    """

    # TODO: LSTM의 hidden dimension configuration @ 소연
    # edit by seyeon
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, all=True, **kwargs):
        if all:
            # LSTM에 전체 문장 token embedding 을 넣을 경우임
            # 일단 현재는 LSTM 사용할 경우 모든 sequence 받는걸로 설정.
            x = features
        else:
            # LSTM으로 들어갈 shape 때문에
            # [CLS] embedding 만 LSTM에 들어간다면 unsqueeze 처리 필요
            # 단, 이것만 LSTM 처리하는건 FC layer 넣는거랑 별반 다를게 없음.
            # 따라서 entity embedding을 더 입력으로 받거나 아니면 위에 처럼
            # 전체 token embedding sequence  넣는 것이 필요
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
            x = x.unsqueeze(dim=1)
        x = self.dropout(x)

        output, _ = self.lstm(x)
        _, _, hidden_size = output.size()
        hidden_size = hidden_size // 2
        forward = output[:, -1, :hidden_size]  # forward LSTM's output
        backward = output[:, 0, hidden_size:]  # backward LSTM's output
        x = torch.cat([forward, backward], dim=-1)
        x = torch.tanh(x)  # activation
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 이거도 수정중
class customRBERTForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # from transformers import RobertaModel

        # 여기서 add_pooling_layer를 추가할지 안할지 선택합니다.
        self.roberta = customRobertaModel(config, add_pooling_layer=False)

        # TODO : 여기서 classification head를 어떤 모드로 할지 분기합니다.

        if config.head_type == "more_dense":
            self.classifier = RobertaClassificationHead(config)
        elif config.head_type =="lstm":
            self.classifier = RobertaClassificationLSTMHead(config)
        elif config.head_type =="modifiedBiLSTM":
            self.classifier = RobertaClassificationBidirectionalLSTMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        # self._backward_compatibility_gradient_checkpointing()

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        _init_weights= True
        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        e1_mask: Optional[torch.Tensor] = None,
        e2_mask: Optional[torch.Tensor] = None,

    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            e1_mask = e1_mask,
            e2_mask = e2_mask

        )

        # TODO: 위에서 정의된 model의 아웃풋으로 나오게 되면, 해당 output을 사용하고
        # sequence_output으로 classifier에 넘기게 됩니다. 이때 어떤 값이 넘어가고, dimension을 갖는지 확인 필요합니다.
        sequence_output = outputs[0]
        # breakpoint()
        logits = self.classifier(sequence_output,
                                 e1_mask = e1_mask,
                                 e2_mask = e2_mask)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 최종 output이 다음과 같은 class로 나가기 때문에,
        # custom loss 사용시 원하는 key, value값 선택해서 적용하면 됩니다.
        # 단, loss가 적용되는 시점을 이 forward 부분말고 Trainer의 compute_loss에서 수행했습니다.
        # 아마 여기서 loss 계산하더라도 compute_loss에서 optimizer.step()가 수행되기 때문에
        # 결국 어떤 loss로 optimize되는지는 거기서 결정되는 것 같습니다.
        # 어떤 차이가 있는지 확인은 필요할듯 @ 소연
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__=='__main__':
    from transformers.modeling_utils import PreTrainedModel
    name = 'roberta-large'
    conf = AutoConfig.from_pretrained(name)
    # breakpoint()
    # model = RobertaModel(conf)
    model =customRobertaForSequenceClassification(conf)

    # model = RobertaForSequenceClassification(conf)

    print(model)